import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import time
from model import *
from tokenizer import *
from copy import deepcopy
import json
import torch.optim.lr_scheduler as lr_scheduler
import os
import gc
from torch.cuda.amp import GradScaler, autocast

# def train_epoch(model, optimizer, criterion, data_loader, device, epoch, scheduler=None):
#     torch.cuda.empty_cache()
#     model.train()
#     total_correct, total_count = 0, 0
#     total_loss = 0
#     log_interval = 10
#     start_time = time.time()

#     for idx, (inputs, labels) in enumerate(data_loader):
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         if scheduler:
#             scheduler.step()

#         total_correct += (outputs.argmax(1) == labels).sum().item()
#         total_count += labels.size(0)
#         total_loss += loss.item()

#         # log results every 10 batches
#         if idx % log_interval == 0 and idx > 0:
#             elapsed = time.time() - start_time
#             print(f'Epoch {epoch+1} | {idx}/{len(data_loader)} batches | '
#                   f'Accuracy: {total_correct / total_count:.3f} | '
#                   f'Loss: {loss.item():.3f} | '
#                   f'Elapsed time: {elapsed:.2f}s')
#             start_time = time.time()
    
#     return total_loss / len(data_loader), total_correct / total_count


def train_epoch(model, optimizer, criterion, data_loader, device, epoch, scheduler=None, grad_accumulation_steps=1):
    torch.cuda.empty_cache()
    model.train()
    scaler = GradScaler()  # Initialize the gradient scaler for mixed precision
    total_correct, total_count = 0, 0
    total_loss = 0
    log_interval = 10
    start_time = time.time()
    optimizer.zero_grad()  # Move optimizer.zero_grad() outside the batch loop

    for idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Start of mixed precision block
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels) / grad_accumulation_steps  # Scale loss

        # Scale loss to prevent underflow
        scaler.scale(loss).backward()

        if (idx + 1) % grad_accumulation_steps == 0:  # Only step every grad_accumulation_steps
            scaler.step(optimizer)  # Adjust model parameters
            scaler.update()  # Update the scale for next iteration
            optimizer.zero_grad()  # Zero the parameter gradients

        if scheduler and (idx + 1) % grad_accumulation_steps == 0:
            scheduler.step()  # Update the learning rate

        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        total_loss += loss.item() * grad_accumulation_steps  # Unscale the loss

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch+1} | {idx}/{len(data_loader)} batches | '
                  f'Accuracy: {total_correct / total_count:.3f} | '
                  f'Loss: {loss.item() * grad_accumulation_steps:.3f} | '  # Unscale the loss for logging
                  f'Elapsed time: {elapsed:.2f}s')
            start_time = time.time()

    # After the epoch, ensure any remaining gradients are processed
    if len(data_loader) % grad_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / len(data_loader), total_correct / total_count

def evaluate(model, data_loader, device, criterion):
    model.eval()
    total_correct, total_count = 0, 0
    total_loss = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item()

    return total_loss / len(data_loader), total_correct / total_count


def train(model, optimizer, criterion, train_loader, valid_loader, num_epochs, save_path, option='val'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stopping = 2
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    for epoch in range(num_epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, device, epoch, scheduler)

        if option == 'val':
            val_loss, val_acc = evaluate(model, valid_loader, device, criterion)
            val_time = time.time() - epoch_start_time

            print(f'Epoch {epoch+1}/{num_epochs} | '
                f'Train Loss: {train_loss:.4f} | '
                f'Train Accuracy: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f} | '
                f'Val Accuracy: {val_acc:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping:
                    print("Early stopping due to no improvement in validation loss.")
                    break

            print(f'End of epoch {epoch+1} | Time: {val_time:.2f}s | Validation accuracy: {val_acc:.3f}')

        else:
            print(f'Epoch {epoch+1}/{num_epochs} | '
                f'Train Loss: {train_loss:.4f} | '
                f'Train Accuracy: {train_acc:.4f} | ')

    torch.cuda.empty_cache()  # Clear GPU cache after evaluation
    gc.collect()  # Collect garbage to free memory

def tune_hyperparameters(config, train_dataset, valid_dataset):
    # Hyperparameters space
    learning_rates = [0.001]
    batch_sizes = [64]
    num_encoder_layers_options = [2,3]
    dim_feedforward_options = [128, 216, 512]
    nhead_options = [4, 8]
    best_val_accuracy = 0.0
    best_config = None
    i = 1


    for batch_size in batch_sizes:
        results = []
        for lr in learning_rates:
            for num_encoder_layers in num_encoder_layers_options:
                for dim_feedforward in dim_feedforward_options:
                    for nhead in nhead_options:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        current_config = deepcopy(config)
                        current_config.lr = lr
                        current_config.batch_size = batch_size
                        current_config.num_encoder_layers = num_encoder_layers
                        current_config.dim_feedforward = dim_feedforward
                        current_config.nhead = nhead

                        # Initialize dataloaders with the current batch size
                        train_loader = DataLoader(train_dataset, batch_size=current_config.batch_size, shuffle=True, collate_fn=collate_batch)
                        valid_loader = DataLoader(valid_dataset, batch_size=current_config.batch_size, collate_fn=collate_batch)

                        # Initialize the model
                        model = TransformerEncoderModel(current_config)

                        if torch.cuda.device_count() > 1:
                            print("Let's use", torch.cuda.device_count(), "GPUs")
                            model = nn.DataParallel(model)

                        model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=current_config.lr)
                        criterion = torch.nn.CrossEntropyLoss()

                        # Train and validate the model
                        config_attrs = vars(current_config)
                        formatted_config = ', '.join(f'{key}: {value}' for key, value in config_attrs.items())
                        print(f"Training with config: {formatted_config}")

                        save_model_path = f'./valid_model/model_{i}'
                        os.makedirs(save_model_path, exist_ok=True)
                        train(model, optimizer, criterion, train_loader, valid_loader, current_config.max_epochs, save_model_path)

                        # Evaluate model on validation set to get the accuracy
                        val_loss, val_accuracy = evaluate(model, valid_loader, device, criterion)

                        print(f'Val Loss: {val_loss:.4f} | '
                                f'Val Accuracy: {val_accuracy:.4f}')

                        # Check if the current hyperparameters are the best
                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            best_config = deepcopy(current_config)

                        # Log the results
                        results.append({
                            'lr': lr,
                            'batch_size': batch_size,
                            'num_encoder_layers': num_encoder_layers,
                            'dim_feedforward': dim_feedforward,
                            'nhead': nhead,
                            'val_accuracy': val_accuracy
                        })

                        i += 1

        # Save the results to a JSON file
        with open(f'hyperparameter_tuning_results_{batch_size}.json', 'w') as f:
            json.dump(results, f, indent=4)


    print(f"Best Validation Accuracy: {best_val_accuracy}")
    print(f"Best Hyperparameters: {vars(best_config)}")

    return best_config