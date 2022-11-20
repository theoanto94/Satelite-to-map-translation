"""DATASET PREPROCESS
We load Normalize and perform data Augmentation to the data"""
from config import *


print("Dataset Preprocess Starts")

#Transform and Augment our Images
data_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.CenterCrop((256, 512)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = datasets.ImageFolder(root=os.path.join(MAPS_PATH,"train"), transform=data_transform)
dataset_val = datasets.ImageFolder(root=os.path.join(MAPS_PATH, "val"), transform=data_transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dataloader_val = torch.utils.data.DataLoader(dataset_val,   batch_size=18, shuffle=True, num_workers=0)

print(f"Training Images: {len(dataset_train)}")
print(f"Validation/Test Images: {len(dataset_val)}")
print("Loading Dataset Finished")


def show_image(img, title="No title", figsize=(5, 5),img_name ="Sample_Image" ):
    '''Fucntion that reverts the Normalization Transformation and Returns a requested image to
    get saved'''
    img = img.numpy().transpose(1, 2, 0)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    img = img * std + mean
    np.clip(img, 0, 1)

    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.savefig(f"{SAVE_PATH}{img_name}")



images,_ = next(iter(dataloader_train))

satelite_images = images[0][:,:,:256]
map_images = images[0][:,:,256:]

show_image(images[0], title="Satellite to Map", figsize=(8,8), img_name= "Sample_Image" )
show_image(img = satelite_images, title = "Satellite Image", figsize=(5,5), img_name="Satelite_Image")
show_image(img = map_images, title = "Map Image", figsize=(5,5), img_name="Map_Image")



