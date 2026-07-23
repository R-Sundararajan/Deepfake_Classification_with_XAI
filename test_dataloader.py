from datasets.dataloader import get_dataloaders

train_loader, valid_loader, test_loader = get_dataloaders()

print("Train batches :", len(train_loader))
print("Validation batches :", len(valid_loader))
print("Test batches :", len(test_loader))