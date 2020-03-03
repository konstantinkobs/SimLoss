import logging
from datetime import datetime
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from ignite.engine import Events, Engine, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError, MeanSquaredError
from ignite.handlers import EarlyStopping

from models.model import LeNet
from models.loss import SimLoss
from data.data_loader import get_train_valid_loader, get_test_loader

from simloss.utils.ignite import PredictionMetric, SuperclassAccuracy, FailedSuperclassAccuracy
from simloss.utils.constants import CIFAR100_CLASSES, CIFAR100_CLASSES_FILTERED, CIFAR100_BLACKLIST, CIFAR_PATH, SIM_MATRIX_PATH

# Make the execution reproducible
# torch.manual_seed(42)
# np.random.seed(42)

best_metric = {'accuracy': 0.0}
best_test_metric = None
best_epoch = 0

@click.command()
@click.option('--epochs', type=click.INT, default=1000)
@click.option('--batchsize', type=click.INT, default=1024)
@click.option('--patience', type=click.INT, default=20)
@click.option('--learningrate', type=click.FLOAT, default=0.001)
@click.option('--lower-bound', type=click.FLOAT, default=0.5)
@click.option('--use-cross-entropy', is_flag=True)
@click.option('--test-class', type=click.Choice(CIFAR100_CLASSES), default=[], multiple=True)
@click.option('--val-class', type=click.Choice(CIFAR100_CLASSES), default=[], multiple=True)
@click.option('--show-samples', is_flag=True)
@click.option('--run', type=click.INT, default=1)
def main(epochs: int,
         batchsize: int,
         learningrate: float,
         lower_bound: float,
         use_cross_entropy: bool,
         test_class: str,
         val_class: str,
         show_samples: bool,
         patience: int,
         run: int) -> None:
    logger = logging.getLogger(__name__)
    logger.info('Start training')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 6
    pin_memory = device == 'cuda' 

    if use_cross_entropy:
        sim_matrix = torch.eye(len(CIFAR100_CLASSES_FILTERED))
    else:
        assert lower_bound < 1.0
        sim_matrix = torch.from_numpy(np.load(SIM_MATRIX_PATH))
        sim_matrix = (sim_matrix - lower_bound) / (1 - lower_bound)
        sim_matrix[sim_matrix < 0.0] = 0.0

    criterion = SimCE(w=sim_matrix)

    train_classes = [c for c in CIFAR100_CLASSES if c not in test_class and 
                                                    c not in val_class and
                                                    c not in CIFAR100_BLACKLIST]

    train_loader, val_loader = get_train_valid_loader(data_dir=CIFAR_PATH,
                                                      batch_size=batchsize,
                                                      augment=True,
                                                      random_seed=None,
                                                      valid_size=0.1,
                                                      shuffle=True,
                                                      show_sample=show_samples,
                                                      num_workers=num_workers,
                                                      pin_memory=pin_memory,
                                                      train_classes=train_classes,
                                                      val_classes=val_class)

    test_loader = get_test_loader(data_dir=CIFAR_PATH,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  test_classes=test_class)

    model = LeNet()
    optimizer = optim.Adam(model.parameters(), lr=learningrate)

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    metrics = {
        'accuracy': Accuracy(),
        'superclass_accuracy': SuperclassAccuracy(),
        'failed_superclass_accuracy': FailedSuperclassAccuracy(),
        'loss': Loss(loss_fn=criterion)
    }

    test_metrics = metrics

    # We have to create multiple evaluators since the early stopping handler is only supposed to use the validation
    # metrics
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    test_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device=device)


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        logger.info(f'Training (Epoch {trainer.state.epoch}): {trainer.state.output:.3f}')


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        met = ', '.join([f'{m}: {train_evaluator.state.metrics[m]:.3f}' for m in train_evaluator.state.metrics])
        logger.info(f'Training (Epoch {trainer.state.epoch}): {met}')


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        global best_epoch, best_metric, best_test_metric

        val_evaluator.run(val_loader)
        met = ', '.join([f'{m}: {val_evaluator.state.metrics[m]:.3f}' for m in val_evaluator.state.metrics])
        logger.info(f'Validation (Epoch {trainer.state.epoch}): {met}')

        if val_evaluator.state.metrics['accuracy'] > best_metric['accuracy']:
            best_metric = val_evaluator.state.metrics
            best_epoch = trainer.state.epoch

            test_evaluator.run(test_loader)
            metrics_str = ', '.join([f'{m}: {test_evaluator.state.metrics[m]: .3f}' for m in test_evaluator.state.metrics])
            logger.info(f'Test: {metrics_str}')

            best_test_metric = test_evaluator.state.metrics

        elif trainer.state.epoch - best_epoch > patience:
            logger.info("Stopping early due to no improvement")
            trainer.terminate()
            
    @trainer.on(Events.COMPLETED)
    def log_test_results(trainer):
        global best_test_metric, best_metric, best_epoch

        print("="*20)
        print(f"Run: {run}")
        print(f"Lower bound: {lower_bound}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best evaluation metrics: {', '.join([f'{m}: {best_metric[m]: .3f}' for m in best_metric])}")
        print(f"Best test metrics: {', '.join([f'{m}: {best_test_metric[m]: .3f}' for m in best_test_metric])}")

    # @trainer.on(Events.COMPLETED)
    # def save_model(trainer):
    #     outfilename = args.outfile.split('/')[-1].split('.')[0]
    #     torch.save(model.state_dict(),
    #                'checkpoints/{}_{}bins_fold{}_{}_{}.p'.format(repr(criterion), args.bins, args.fold, args.dataset,
    #                                                              outfilename))

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[1]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
