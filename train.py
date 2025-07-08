from ultralytics import YOLO
import argparse
# Model configuration file
model_yaml_path = "ultralytics/cfg/models/..."
# Dataset configuration file
data_yaml_path = 'ultralytics/cfg/datasets/...'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train or validate YOLO model.')
# train is used to train the base model, while val is used to obtain accuracy metrics.
parser.add_argument('--mode', type=str, default='train', help='Mode of operation.')
# Dataset storage path
parser.add_argument('--data', type=str, default='model_yaml_path', help='Path to data file.')
parser.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
parser.add_argument('--batch', type=int, default=39, help='Batch size.')
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'], default='SGD', help='optimizer')
parser.add_argument('--workers', type=int, default=3, help='Number of workers.')
parser.add_argument('--device', type=str, default='0', help='Device to use.')
parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
parser.add_argument('--close-mosaic', type=int, default=0, help='Experimental')
parser.add_argument('--seed', type=int, default=0, help='Global training seed')
args = parser.parse_args()


def train(model, data, epoch, batch, workers, device, name):
    model.train(data=data, epochs=epoch, batch=batch, workers=workers, device=device, name=name)


def validate(model, data, batch, workers, device, name):
    model.val(data=data, batch=batch, workers=workers, device=device, name=name)


def main():
    model = YOLO(args.weights)
    if args.mode == 'train':
        train(model, args.data, args.epoch, args.batch, args.workers, args.device, args.name)
    else:
        validate(model, args.data, args.batch, args.workers, args.device, args.name)


if __name__ == '__main__':
    main()
