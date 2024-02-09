mkdir imagenet_data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mv ILSVRC2012_img_val.tar imagenet_data/
cd imagenet_data
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ../
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
tar xzf ILSVRC2012_devkit_t12.tar.gz
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt