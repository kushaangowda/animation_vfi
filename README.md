[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ifbeTrPr)
# e6691-2024Spring-project

For general requiremenets, refer to the [project instruction document](https://docs.google.com/document/d/1IqkNFUTRoI8xk0a-xawlIzA_QHHk5pZcRZY-q-zey1Q/edit?usp=share_link).


export LD_LIBRARY_PATH=""

torchrun --nproc_per_node=1 main.py --batch_size=16 --epochs=50 --data_path="image_data.h5" --mode="train" --type=1 --path=""

torchrun --nproc_per_node=1 main.py --batch_size=1 --data_path="image_data.h5" --mode="predict" --type=1 --path="good_model_resnet_percloss_60ssim.pth"