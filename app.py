import os
from flask import Flask, render_template, request
from predictor import check


author = 'TEAM DELTA'

app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
@app.route('/index')
def index():
    return render_template('upload.html')
# class ConvBlock(nn.Module):
def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def forward(self, x):
    x = self.conv(x)
    return x

# class UpConv(nn.Module):
def __init__(self, in_channels, out_channels, bilinear=True):
    super(UpConv, self).__init__()

    if bilinear:
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    else:
        self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)

    self.conv = ConvBlock(in_channels, out_channels)

def forward(self, x1, x2):
    x1 = self.up(x1)
    x = torch.cat([x2, x1], dim=1)
    x = self.conv(x)
    return x

# class AttentionGate(nn.Module):
def __init__(self, in_channels):
    super(AttentionGate, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
        nn.BatchNorm2d(in_channels//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels//2, 1, kernel_size=1),
        nn.Sigmoid()
    )

def forward(self, x):
    g = F.avg_pool2d(x, kernel_size=x.size()[2:]) # Global Average Pooling
    g = self.conv(g)
    x = x * g
    return x

# class ResUNet(nn.Module):
def __init__(self, in_channels=3, out_channels=1, bilinear=True):
    super(ResUNet, self).__init__()
    self.bilinear = bilinear

    self.down1 = ConvBlock(in_channels, 64)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.down2 = ConvBlock(64, 128)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.down3 = ConvBlock(128, 256)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.down4 = ConvBlock(256, 512)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.center = ConvBlock(512, 1024)

    self.up4 = UpConv(1024, 512, bilinear)
    self.attention4 = AttentionGate(512)
    self.up3 = UpConv(512, 256, bilinear)
    self.attention3 = AttentionGate(256)
    self.up2 = UpConv(256, 128, bilinear)
    self.attention2 = AttentionGate(128)
    self.up1 = UpConv(128, 64, bilinear)
    self.attention1 = AttentionGate(64)

    self.final = nn.Conv2d

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        print(filename)
        dest = '/'.join([target, filename])
        print(dest)
        file.save(dest)

    status = check(filename)

    return render_template('complete.html', image_name=filename, predvalue=status)

if __name__ == "main":
    app.run(port=4555, debug=True)
