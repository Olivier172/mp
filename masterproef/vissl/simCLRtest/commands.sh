#get weights resnet 50
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -P content/

mkdir -p content/dummy_data/train/class1;
mkdir -p content/dummy_data/train/class2;
mkdir -p content/dummy_data/val/class1;
mkdir -p content/dummy_data/val/class2;

# create 2 classes in train and add 5 images per class
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class1/img1.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class1/img2.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class1/img3.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class1/img4.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class1/img5.jpg;

wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class2/img1.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class2/img2.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class2/img3.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class2/img4.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/train/class2/img5.jpg;

# create 2 classes in val and add 5 images per class
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class1/img1.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class1/img2.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class1/img3.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class1/img4.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class1/img5.jpg;

wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class2/img1.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class2/img2.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class2/img3.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class2/img4.jpg;
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O content/dummy_data/val/class2/img5.jpg;
