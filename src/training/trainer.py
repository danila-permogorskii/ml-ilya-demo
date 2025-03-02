"""
Training utility for managing the training process of neural networks.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    """
    Trainer class for managing model training and evaluation.
    
    This class handles the training loop, model checkpointing, 
    performance tracking, and evaluation.
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 criterion=None, optimizer=None, scheduler=None,
                 device=None, checkpoint_dir='checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): PyTorch model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            criterion (nn.Module, optional): Loss function
            optimizer (optim.Optimizer, optional): Optimizer
            scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
            device (torch.device, optional): Device to use for training
            checkpoint_dir (str): Directory to save checkpoints
        """
        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
            
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
            
        # Learning rate scheduler
        self.scheduler = scheduler
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc='Training')
        
        for inputs, targets in pbar:
            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
            
        # Calculate epoch statistics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate validation statistics
        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, epochs, save_best=True, early_stopping=None):
        """
        Train the model for multiple epochs.
        
        Args:
            epochs (int): Number of epochs to train
            save_best (bool): Whether to save the best model
            early_stopping (int, optional): Early stopping patience
            
        Returns:
            dict: Training history
        """
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(f"best_model.pth")
                print(f"New best model saved with accuracy: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping is not None and patience_counter >= early_stopping:
                print(f"Early stopping after {epoch} epochs")
                break
                
            # Save checkpoint at regular intervals
            if epoch % 5 == 0 or epoch == epochs:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
                
        # Training complete
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Return training history
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accuracies,
            'val_acc': self.val_accuracies
        }
    
    def save_checkpoint(self, filename):
        """
        Save a model checkpoint.
        
        Args:
            filename (str): Name of the checkpoint file
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)
        
    def load_checkpoint(self, filename):
        """
        Load a model checkpoint.
        
        Args:
            filename (str): Name of the checkpoint file
            
        Returns:
            dict: Checkpoint data
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        return checkpoint
    
    def plot_history(self, figsize=(12, 5)):
        """
        Plot training history.
        
        Args:
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy Curves')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_history.png'))
        plt.show()
        
    def predict(self, inputs):
        """
        Make predictions with the model.
        
        Args:
            inputs (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predicted classes
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
        return predicted
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on a test set.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            tuple: (accuracy, class_accuracies, confusion_matrix)
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Testing'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        # Overall accuracy
        accuracy = 100 * np.mean(all_predictions == all_targets)
        
        # Per-class accuracy
        classes = np.unique(all_targets)
        class_accuracies = {}
        
        for c in classes:
            mask = all_targets == c
            class_acc = 100 * np.mean(all_predictions[mask] == all_targets[mask])
            class_accuracies[int(c)] = class_acc
        
        # Confusion matrix
        conf_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for i in range(len(all_targets)):
            conf_matrix[all_targets[i], all_predictions[i]] += 1
            
        return accuracy, class_accuracies, conf_matrix
