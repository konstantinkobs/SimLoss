import numpy as np
from dataset import TrainValTestDataset, transforms
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
from loss import SimLoss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError, MeanSquaredError
import argparse
from model import Model
from torchvision.datasets import ImageFolder

# Make the execution reproducible
# torch.manual_seed(42)
# np.random.seed(42)

number_of_classes = 90  # 61

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", default=200, type=int, help="number of training epochs")
parser.add_argument("--stopping", default=10, type=int, help="number of epochs the early stopping waits until it terminates the training")
parser.add_argument("--batchsize", default=1024, type=int, help="batch size")
parser.add_argument("--learningrate", default=0.001, type=float, help="learning rate")
parser.add_argument("--reductionfactor", default=0.0, type=float,
                    help="Reduction factor for SimLoss")
parser.add_argument("--number", default=0, type=int, help="index of the run to get the correct run")
parser.add_argument("--dataset", default="AFAD", type=str, help="name of dataset")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 6

criterion = SimLoss(number_of_classes, args.reductionfactor, device)

image_dataset = ImageFolder("data/" + args.dataset, transform=transforms)

train_dataset = TrainValTestDataset(image_dataset, mode="train")
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batchsize,
                          shuffle=True,
                          num_workers=num_workers)

val_dataset = TrainValTestDataset(image_dataset, mode="validate")
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=args.batchsize,
                        num_workers=num_workers)

test_dataset = TrainValTestDataset(image_dataset, mode="test")
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batchsize,
                         num_workers=num_workers)

model = Model(number_of_classes=number_of_classes)

optimizer = optim.Adam(model.parameters(), lr=args.learningrate)

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

metrics = {
    "accuracy": Accuracy(),
    "MAE": MeanAbsoluteError(output_transform=lambda out: (torch.max(out[0], dim=1)[1], out[1])),
    "MSE": MeanSquaredError(output_transform=lambda out: (torch.max(out[0], dim=1)[1], out[1])),
    "loss": Loss(loss_fn=criterion)
}

evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    print(f"Training (Epoch {trainer.state.epoch}): {trainer.state.output:.3f}")


best_epoch = 0
best_val_metrics = {"MAE": np.inf}
best_test_metrics = {"MAE": np.inf}


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    global best_epoch, best_val_metrics, best_test_metrics

    evaluator.run(val_loader)
    met = ', '.join([f'{m}: {evaluator.state.metrics[m]:.3f}' for m in evaluator.state.metrics])
    print(f"Validation (Epoch {trainer.state.epoch}): {met}")

    if evaluator.state.metrics["MAE"] < best_val_metrics["MAE"]:
        best_val_metrics = evaluator.state.metrics
        best_epoch = trainer.state.epoch

        evaluator.run(test_loader)
        best_test_metrics = evaluator.state.metrics

        met = ', '.join([f'{m}: {evaluator.state.metrics[m]:.3f}' for m in evaluator.state.metrics])
        print(f"Test (Epoch {trainer.state.epoch}): {met}")

        save_model()

    if trainer.state.epoch - best_epoch > args.stopping:
        print("Terminating training due to no improvement")
        trainer.terminate()


@trainer.on(Events.COMPLETED)
def log_test_results(trainer):
    global best_epoch, best_val_metrics, best_test_metrics

    print("="*30)
    print(f"RUN:             {args.number:02d}")
    print(f"REDUCTIONFACTOR: {args.reductionfactor}")
    print(f"BEST EPOCH:      {best_epoch}")
    print(f"BEST VALIDATION: {', '.join([f'{m}: {best_val_metrics[m]:.5f}' for m in best_val_metrics])}")
    print(f"BEST TEST:       {', '.join([f'{m}: {best_test_metrics[m]:.5f}' for m in best_test_metrics])}")


def save_model():
    torch.save(model.state_dict(), f"checkpoints_{args.dataset}/model_{args.reductionfactor}_{args.number:02d}.p")


trainer.run(train_loader, max_epochs=args.epochs)
