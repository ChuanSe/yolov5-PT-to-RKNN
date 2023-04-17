## <div align="center">yolov5模型(.pt)在RK3588(S)上的部署(实时摄像头检测)</div>

- 所需：
  - 安装了Ubuntu20系统的RK3588
  - 安装了Ubuntu18的电脑或者虚拟机
<details>
<summary>一、yolov5 PT模型获取</summary>

[Anaconda教程](https://blog.csdn.net/qq_25033587/article/details/89377259)\
[YOLOv5教程](https://zhuanlan.zhihu.com/p/501798155)\
经过上面两个教程之后，你应该获取了自己的`best.pt`文件

</details>
<details>
<summary>二、PT模型转onnx模型</summary>

- 将`models/yolo.py`文件中的`class`类下的`forward`函数由：
```python
def forward(self, x):
    z = []  # inference output
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if not self.training:  # inference
            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            if isinstance(self, Segment):  # (boxes + masks)
                xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
            else:  # Detect (boxes only)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))
    return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
```
改为：
```python
def forward(self, x):
    z = []  # inference output
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
    return x
```
- 将`export.py`文件中的`run`函数下的语句：
```python
shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
```
改为：
```python
shape = tuple((y[0] if isinstance(y, tuple) else y))  # model output shape
```
- 将你训练模型对应的`run/train/`目录下的`exp/weighst/best.pt`文件移动至与`detect.py`同目录下
- 保证工作目录位于yolov5主文件夹,在控制台执行语句：
```bash
cd yolov5 
python export.py --weights best.pt --img 640 --batch 1 --include onnx --opset 12
```
- 然后在主文件夹下出现了一个`best.onnx`文件，在[Netron](https://netron.app/)中查看模型是否正确
- 点击左上角菜单->Properties...
- 查看右侧`OUTPUTS`是否出现三个输出节点，是则ONNX模型转换成功。
- 如果转换好的`best.onnx`模型不是三个输出节点，则不用尝试下一步，会各种报错。
</details>
<details open>
<summary>三、onnx模型转rknn模型</summary>

- 我使用的是`VMWare`虚拟机安装的`Ubuntu18.04`系统，注意，不是在`RK3588`上，是在你的电脑或者虚拟机上操作这一步骤。
- `rknn-toolkit2-1.4.0`所需`python`版本为`3.6`所以需要安装`Miniconda`来帮助管理。
- 安装[`Miniconda for Linux`](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
  - 进入到下载得到的`Miniconda3-latest-Linux-x86_64.sh`所在目录
    ```bash
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    ```
  - 提示什么都一直同意，直到安装完毕。
  - 安装成功后，重新打开终端。
  - 如果安装成功，终端最前面应该会有一个`(base)`
  - 安装失败的去参考别的`Miniconda3`安装教程。
  - 创建虚拟环境:
      ```bash
      conda create -n rknn3.6 python=3.6 
      ```
  - 激活虚拟环境:
      ```bash
      conda activate rknn3.6
      ```
  - 激活成功时，终端最前面应该会有一个`(rknn3.6)`
- 下载[`rknn-toolkit2-1.4.0`](https://www.t-firefly.com/doc/download/164.html)
  - 到Ubuntu,下载`源代码`下的`RK356X/RK3588 RKNN SDK`
  - 进入百度网盘：`RKNN_SDK-> RK_NPU_SDK_1.4.0` 下载 `rknn-toolkit2-1.4.0` 
  - 下载到Ubuntu后，进入`rknn-toolkit2-1.4.0`目录
    ```bash
    pip install packages/rknn_toolkit2-1.4.0_22dcfef4-cp36-cp36m-linux_x86_64.whl 
    ```
  - 等待安装完毕检查是否安装成功：
    ```bash
    python
    from rknn.api import RKNN
    ```
  - 如果没有报错则成功。
  - 如果报错：
    - 1.是否处于`rknn3.6`虚拟环境下；
    - 2.`pip install packages/rknn_toolkit2-1.4.0_22dcfef4-cp36-cp36m-linux_x86_64.whl`是否报错;
    - 3.`pip install`报错的时候，提示缺什么就用`pip install`或者`sudo apt-get install`安装什么;

- 上述所需都安装并且验证成功，则开始下一步。
- 将`best.onnx`模型转换为`best.rknn`模型
  - 进入转换目录：
    ```bash
    cd examples/onnx/yolov5
    ```
  - 最好是复制一份`test.py`出来进行修改：
    ```bash
    cp test.py ./mytest.py
    ```
  - 将一开始定义的文件进行修改,这是我修改之后的：
    ```python
    ONNX_MODEL = 'best.onnx'    #待转换的onnx模型
    RKNN_MODEL = 'best.rknn'    #转换后的rknn模型
    IMG_PATH = './1.jpg'        #用于测试图片
    DATASET = './dataset.txt'   #用于测试的数据集，内容为多个测试图片的名字
    QUANTIZE_ON = True          #不修改
    OBJ_THRESH = 0.25           #不修改
    NMS_THRESH = 0.45           #不修改
    IMG_SIZE = 640              #不修改
    CLASSES = ("person")        #修改为你所训练的模型所含的标签
    ```
  - 将`if __name__ == '__main__':`中的两个语句：
    ```python
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    ```
  - 修改为
    ```python
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
    ```
  - 想要程序执行完，展示推理效果，将以下语句：
    ```python
    # cv2.imshow("post process result", img_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ```
  - 注释打开：
    ```python
    cv2.imshow("post process result", img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```
  - 终端执行：
    ```bash
    python mytest.py
    ```
  - 运行完展示效果，以及文件夹中出现`best.rknn`则该步骤成功。  
</details>
<details open>
<summary>四、在RKNN3588上部署rknn模型并实时摄像头推理检测</summary>

- 在`RKNN3588`的`Ubuntu20`系统上安装`Miniconda`，需要注意的是，`RKNN3588`的`Ubuntu20`系统为`aarch`架构因此下载的`Miniconda`版本和之前有所不同，需要选择对应的`aarch`版本。
- [`aarchMiniconda下载`](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh)
- 安装不再赘述。
- 创建虚拟环境,因为在`RK3588`上要用到`rknn-toolkit-lite2`所以需要安装`python3.7`:
  - conda create -n rknnlite3.7 python=3.7
  - conda activate rknnlite3.7
- 下载`rknn-toolkit-lite2`到`RK3588`，也就是下载[`rknn-toolkit2-1.4.0`](https://www.t-firefly.com/doc/download/164.html)，不再赘述。
- 安装`rknn-toolkit-lite2`
  - 进入`rknn-toolkit2-1.4.0/rknn-toolkit-lite2`目录
    ```bash
    pip install packages/rknn_toolkit_lite2-1.4.0-cp37-cp37m-linux_aarch64.whl
    ```
  - 等待安装完毕
  - 测试是否安装成功：
    ```bash
    python
    from rknnlite.api import RKNNLite
    ```
  - 不报错则成功
- 在`example`文件夹下新建一个`test`文件夹
- 在其中放入你转换成功的`best.rknn`模型以及该`github`仓库下的`detect.py`文件
- `detect.py`文件中需要修改的地方：
  - 定义
    ```python
    RKNN_MODEL = 'best.rknn'      #你的模型名称
    IMG_PATH = './1.jpg'          #测试图片名
    CLASSES = ("cap")             #标签名
    ```
  - `if __name__ == '__main__':`:
    ```python
    capture = cv2.VideoCapture(11)      #其中的数字为你Webcam的设备编号
    ```
    - 关于设备编号，在终端中运行： 
      ```bash
      v4l2-ctl --list-devices     
      ```
    - 打印出的`Cam`之类的字眼对应的`/dev/video11`中的11就是你的设备编号。  
- 运行脚本：
  ```bash
  python detect.py
  ```
- 部署完成。
- 有任何疑问可以发送至邮箱：Chuan5445520@163.com  
</details>
