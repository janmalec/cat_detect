# Import the necessary modules
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from PIL import Image
import cv2
from datetime import datetime
import os
import emoji
import time
import requests

def check_cat(frame, filter='cat&dog'):
    # Load the pre-trained model (ResNet-18) and set it to evaluation mode
    #model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # try ResNet-50
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()

    # Define the transform to resize and normalize the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the image from the file name given by argparse and apply the transform
    image = Image.fromarray(frame)
    image = transform(image)

    # Add a batch dimension and move the image to the device (CPU or GPU)
    image = image.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    # Make a prediction using the model
    output = model(image)

    # Get the index of the class with the highest probability
    _, pred = torch.max(output, 1)


    # Get the class name from the index using the ImageNet labels
    imagenet_labels = torchvision.datasets.utils.download_url('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt', '.', )
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # Write top matches with probabilities in a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    logfile.write('\U0001F4C5 ' + timestamp + '\n')
    # look up top five matches and print them in the log file
    N = 5
    topN = torch.topk(output, N)
    for i in range(N):
        logfile.write(f'{emoji.emojize(":paw_prints:")} {classes[topN.indices[0][i]]}: {topN.values[0][i].item():.2%}\n')
    logfile.write('\n')
    # flush
    logfile.flush()

    # Print the result
    print(f'The top predicted class is: {classes[topN.indices[0][0]]}')

    if filter == 'cat':
        # if 'cat' in top 5 classes, return True
        if any('cat' in classes[topN.indices[0][i]] for i in range(N)):
            return True
    else:
        # List of keywords for breeds of cats and dogs
        keywords = [
            # cat breeds
            'egyptian cat', 'tiger cat', 'persian cat', 'siamese cat', 'tabby', 'lynx', 
            'leopard', 'jaguar', 'lion', 'tiger', 'cheetah', 
            # dog breeds that may look similar to cats
            'pomeranian', 'papillon', 'maltese dog', 'chihuahua', 'japanese spaniel', 'pekinese', 'shih-tzu',
            'border', 'collie', 'husky', 'dog', 'groenendael',
            # other animals that may look similar to cats
            'hog'
        ]

        # if any of the top 5 classes contain a cat/dog breed keyword, return True
        if any(any(keyword in classes[topN.indices[0][i]] for keyword in keywords) for i in range(N)):
            return True
    
    return False



# Function to detect motion
def detect_motion(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        else:
            # Write log message with timestamp and motion detection area
            logfile.write(f'{datetime.now().strftime("%Y%m%d_%H%M%S")} {emoji.emojize(":cat_face:")} Motion detected with {cv2.contourArea(contour)} \n')
            # flush
            logfile.flush()
            return True
    return False

if __name__ == '__main__':
    # Parse the command-line arguments
    # Define the parser
    parser = argparse.ArgumentParser(description="Detect motion and check if a cat is present")
    parser.add_argument('--rtsp_url', type=str, help='RTSP URL for the camera stream')
    parser.add_argument('--topic', type=str, help='topic for ntfy')

    # Parse the arguments
    args = parser.parse_args()

    logfile = open('log.txt', 'a')

    # Open the video
    cap = cv2.VideoCapture(args.rtsp_url)

    try:
        _, frame1 = cap.read()
        _, frame2 = cap.read()
    except cv2.error as e:
        print(f'Error reading frames from camera: {e}')
        logfile.write(f'{datetime.now().strftime("%Y%m%d_%H%M%S")} Error reading frames from camera: {e}\n')
        # flush
        logfile.flush()
    check_cat(frame1)

    while cap.isOpened():
        motion_detected = detect_motion(frame1, frame2)
        timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")

        if motion_detected:
            cv2.imwrite(f'motion_pics/{timestamp}.jpg', frame2)
            if check_cat(frame2):
                cv2.imwrite(f'cats/cat_detected_{timestamp}.jpg', frame2)
                print("Cat detected and image saved!")
                # ntfy alert with the image using requests if topic is provided
                if args.topic:
                    try:
                        with open(f'cats/cat_detected_{timestamp}.jpg', 'rb') as f:
                            r = requests.post(f'https://ntfy.sh/{args.topic}', 
                                                #data = {'title': 'Cat detected 🐈'},
                                                data = f,
                                                headers={'Filename': f'cat_detected_{timestamp}.jpg'})
                            r.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        print(f'Error sending notification: {e}')
                        logfile.write(f'{datetime.now().strftime("%Y%m%d_%H:%M:%S")} Error sending notification: {e}\n')
                    else:
                        print(f'Notification sent: {r.text}')
                        logfile.write(f'{datetime.now().strftime("%Y%m%d_%H%:M%:S")} {emoji.emojize(":bell:")} Notification sent\n')
                    finally:
                        # flush
                        logfile.flush()
                # sleep 5 seconds
                time.sleep(5)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()