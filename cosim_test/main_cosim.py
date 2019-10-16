import torchvision.models as models
import torch
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse

def get_models(model):
    model_list = []

    for i in model.children():
        if i.__class__.__name__ == 'Sequential':
            seq_1st_list = []
            for x in i:
                for y in x.children():
                    if y.__class__.__name__ == 'Sequential':
                        seq_2nd_list = []
                        for z in y:
                            seq_2nd_list.append(z)
                        seq_1st_list.append(tuple(seq_2nd_list))
                    else:
                        seq_1st_list.append(y)
            model_list.append(seq_1st_list)
        else:
            model_list.append(i)
    return model_list

def compare_result(cpu_data, sycl_data, model):
    if torch.allclose(cpu_data, sycl_data.cpu(), rtol=1e-5, atol=1e-5):
        print("model is:", model, "passed")
    else:
        print("model is:", model, "failed!!!")
        print("cpu result")
        print(cpu_data[0][:2])
        print("sycl result")
        print(sycl_data.cpu()[0][:2])
        raise ValueError("Large difference between cpu with sycl") 

def cosim_test(models_list, input_data):
    for model in models_list[:-1]:
        if type(model) is list:
            downsample_input = input_data
            for seq_1st_model in model:
                if type(seq_1st_model) is tuple:
                    for seq_2nd_model in seq_1st_model:
                        cpu_seq_2nd_model = seq_2nd_model.float()
                        cpu_downsample_output = cpu_seq_2nd_model(downsample_input)

                        sycl_seq_2nd_model = seq_2nd_model.sycl()
                        downsample_output = sycl_seq_2nd_model(downsample_input.to('sycl'))

                        downsample_input = cpu_downsample_output
                        # compare cpu and sycl result
                        compare_result(cpu_downsample_output, downsample_output, seq_2nd_model)

                    cpu_output_data += cpu_downsample_output
                    output_data += downsample_output
                else:
                    cpu_seq_1st_model = seq_1st_model.float()
                    cpu_output_data = cpu_seq_1st_model(input_data)

                    sycl_seq_1st_model = seq_1st_model.sycl()
                    output_data = sycl_seq_1st_model(input_data.to('sycl'))
                    # compare cpu and sycl result
                    compare_result(cpu_output_data, output_data, seq_1st_model)

                input_data = cpu_output_data
        else:
            cpu_model = model.float()
            cpu_output = cpu_model(input_data)

            sycl_model = model.sycl()
            output_data = sycl_model(input_data.to('sycl'))

            input_data = cpu_output
            # compare cpu and sycl result
            compare_result(cpu_output, output_data, model)
    if models_list[-1].__class__.__name__ == "Linear":
        input_data = torch.flatten(input_data, 1)

        cpu_model = models_list[-1].float()
        cpu_output = cpu_model(input_data)

        model = models_list[-1].sycl()
        output_data = model(input_data.to('sycl'))
        # compare cpu and sycl result
        compare_result(cpu_output, output_data, models_list[-1])

def main():
    parser = argparse.ArgumentParser(description="do Cosim Test")
    parser.add_argument("-m", "--model", default='resnet50', type=str)
    parser.add_argument("-i", "--iters", default=10, type=int)
    parser.add_argument("-p", "--dataset-path", default="/lustre/dataset/imagenet/img", type=str)
    
    args = parser.parse_args()

    # get model
    model = models.__dict__[args.model]()
    print("get model: {0}".format(args.model))

    # get dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    traindir = os.path.join(args.dataset_path, 'train')
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)

    # cosim loop
    index = 0
    model_list = get_models(model)
    print("get all model layer")

    for input_data, target in train_loader:
        cosim_test(model_list, input_data)
        index += 1
        if index == args.iters:
            break

main()
