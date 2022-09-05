import fedml
from fedml import FedMLRunner
from fedml.data.cifar10.data_loader import load_cifar10_data ,load_partition_data_cifar10
from fedml.data.cifar100.data_loader import load_cifar100_data ,load_partition_data_cifar100
from fedml.model.cv.resnet_gn import resnet18
from trainer import ImageSkippingTrainer


def load_data(args):
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
    if args.dataset == "cifar10":
            data_loader = load_partition_data_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_cifar100
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = data_loader(
        args.dataset,
        args.data_dir,
        args.partition_method,
        args.partition_alpha,
        args.client_num_in_total,
        args.batch_size,
    )
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = resnet18

    client_trainer  = ImageSkippingTrainer(model = model, args=args)
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, client_trainer)
    fedml_runner.run()