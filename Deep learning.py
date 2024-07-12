import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as albu
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

x_train_dir = "C:/Users/SUEN/Desktop/Nazif_PROJE/DATA/mysplitteddata/train/X"
y_train_dir = "C:/Users/SUEN/Desktop/Nazif_PROJE/DATA/mysplitteddata/train/y"

x_valid_dir = "C:/Users/SUEN/Desktop/Nazif_PROJE/DATA/mysplitteddata/validation/X"
y_valid_dir = "C:/Users/SUEN/Desktop/Nazif_PROJE/DATA/mysplitteddata/validation/y"

x_test_dir = "C:/Users/SUEN/Desktop/Nazif_PROJE/DATA/mysplitteddata/test/X"
y_test_dir = "C:/Users/SUEN/Desktop/Nazif_PROJE/DATA/mysplitteddata/test/y"

output_base = 'C:/Users/SUEN/Desktop/oout'

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

class Dataset(BaseDataset):

    
    CLASSES = ['building']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. buildings)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)



dataset = Dataset(x_train_dir, y_train_dir, classes=['building'])
# get some sample
for i in range(1):
    image, mask = dataset[i] 
    visualize(
        image=image, 
        building_mask=mask.squeeze(),
    )


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)





import ssl
ssl._create_default_https_context = ssl._create_unverified_context


ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['building']
BATCH_SIZE = 4
LR = 0.0001

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # could be None for logits or 'softmax2d' for multiclass segmentation
ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'

#Models: Unet, UnetPlusPlus, FPN, PSPNet, DeepLabV3Plus
# create segmentation model with pretrained encoder
model_name = 'Unet'
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)
#number of epoch
start_epoch = 0
new_epoch = 1 
total_epoch =  start_epoch + new_epoch


predict_path = output_base + '/' + model_name + '_{}epoch'.format(total_epoch) + '/prediction'
conf_path = output_base + '/' + model_name + '_{}epoch'.format(total_epoch)+ '/confusion'
model_path = output_base + '/' + model_name + '_{}epoch'.format(total_epoch)
 
if not os.path.exists(model_path):
    os.mkdir(model_path)

if not os.path.exists(predict_path):
    os.mkdir(predict_path)

if not os.path.exists(conf_path):
    os.mkdir(conf_path)

weight_path = model_path + '/' + '/weights'

if not os.path.exists(weight_path):
    os.mkdir(weight_path)

final_model_path= '{}/weights.best_{}_{}_epoch.pth'.format(weight_path,model_name,total_epoch)
best_model_path = '{}/weights.final_{}_{}_epoch.pth'.format(weight_path,model_name,total_epoch)

#Monitoring Model Performance
TRAINING = True 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
loss = smp.utils.losses.DiceLoss()#mode="binary", classes=CLASSES

# Monitoring Model Training
metrics = [smp.utils.metrics.Accuracy(), 
           smp.utils.metrics.IoU(threshold=0.5),
           smp.utils.metrics.Fscore(threshold=0.5), 
           smp.utils.metrics.Precision(), 
           smp.utils.metrics.Recall()]

# define optimizer
#optimizer can also be Adam instead of RMSprop
optimizer = torch.optim.RMSprop([dict(params=model.parameters(), lr=LR),])

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5,)


# Dataset for train images 
train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# sending torch
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)
def train(start_epochs, n_epochs, model, final_model_path, best_model_path):

    # train model
    if TRAINING:

        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for epoch in range(start_epochs, n_epochs):

            # Perform training & validation
            print('\nEpoch: {}'.format(epoch+1))
            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(valid_dataloader)
            # print(valid_logs['accuracy'], valid_logs['iou_score'],valid_logs['fscore'],valid_logs['precision'],valid_logs['recall'])

            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)


            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, best_model_path)
                print('Best Model saved!')

            if epoch == n_epochs-1:
                torch.save(model, final_model_path)
                print('Final Model saved!')
    return model, train_logs_list, valid_logs_list


trained_model,train_logs_list,valid_logs_list = train(start_epoch, total_epoch, model, final_model_path, best_model_path)

train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)
train_logs_df = train_logs_df.set_axis(["loss", "accuracy", "iou_score", "fscore", "precision", "recall"], axis=1, inplace=False)
valid_logs_df = valid_logs_df.set_axis(["val_loss", "val_accuracy", "val_iou_score", "val_fscore", "val_precision", "val_recall"], axis=1, inplace=False)

logs_df = pd.concat([train_logs_df, valid_logs_df], axis=1)

logs_df.to_csv(model_path + "/train_logs.csv", sep=',')

# Plot training & validation iou_score values
plt.figure(figsize=(16, 4))
plt.subplot(1,2,1)
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist())
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.val_iou_score.tolist())
plt.title('Model iou_score')
plt.ylabel('Iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1,2,2)
plt.plot(train_logs_df.index.tolist(), train_logs_df.loss.tolist())
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.val_loss.tolist())
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation precision values
plt.figure(figsize=(16, 4))
plt.subplot(1,2,1)
plt.plot(train_logs_df.index.tolist(), train_logs_df.precision.tolist())
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.val_precision.tolist())
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation recall values
plt.subplot(1,2,2)
plt.plot(train_logs_df.index.tolist(), train_logs_df.recall.tolist())
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.val_recall.tolist())
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation f1-score values
plt.figure(figsize=(16, 4))
plt.subplot(1,2,1)
plt.plot(train_logs_df.index.tolist(), train_logs_df.fscore.tolist())
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.val_fscore.tolist())
plt.title('Model f1-score')
plt.ylabel('F1-score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation accuracy values
plt.subplot(1,2,2)
plt.plot(train_logs_df.index.tolist(), train_logs_df.accuracy.tolist())
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.val_accuracy.tolist())
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.savefig(model_path+r'/metric_graph.png')


#Model Evaluation

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)
test_dataloader = DataLoader(test_dataset)

# load best weights
model = torch.load(final_model_path, map_location=DEVICE)

# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

valid_logs = test_epoch.run(test_dataloader)
metric_name = ['accuracy', 'iou_score', 'fscore', 'precision', 'recall']

print(f"Loss: {valid_logs['dice_loss']:.5f}")
print(f"mean accuracy: {valid_logs['accuracy']:.5f}")
print(f"mean iou_score: {valid_logs['iou_score']:.5f}")
print(f"mean fscore: {valid_logs['fscore']:.5f}")
print(f"mean precision: {valid_logs['precision']:.5f}")
print(f"mean recall: {valid_logs['recall']:.5f}")

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=['building'],
)
test_list = os.listdir(x_test_dir)

for i in range(len(test_list)): # len(test_list)
    image_vis = test_dataset_vis[i][0].astype('uint8')
    image, gt_mask = test_dataset[i]
    
    gt_mask = gt_mask.squeeze()
        
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask,
    )
            
    cv2.imwrite(predict_path + '/'+ test_list[i], pr_mask)


