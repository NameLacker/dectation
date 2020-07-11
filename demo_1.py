'''
Scalar（折线图）组件的接口:
add_scalar(tag, value, step, walltime=None)
接口参数说明：
参数        格式                        含义
tag         string          记录指标的标志，如train/loss，不能含有%
value	    float	        要记录的数据值
step	    int	            记录的步数
walltime	int	            记录数据的时间戳，默认为当前时间戳
'''


'''
from visualdl import LogWriter

if __name__ == '__main__':
    value_1 = [i/1000.0 for i in range(1000)]
    #初始化一个记录器
    with LogWriter(logdir="./log/scalar_test1") as writer:
        for step in range(1000):
            #向记录器添加一个tag为'acc'的数据
            writer.add_scalar(tag="train/acc", step=step, value=value_1[step])
            #向记录器添加一个tag为'loss'的数据
            writer.add_scalar(tag="train/loss", step=step, value=1/(value_1[step] + 1))
    
    value_2 = [i/500.0 for i in range(1000)]
    with LogWriter(logdir="./log/scalar_test2") as writer:
        for step in range(1000):
            writer.add_scalar(tag="train/acc", step=step, value=value_2[step])
            writer.add_scalar(tag="train/loss", step=step, value=1/(value_2[step] + 1))
'''

'''
image组件的记录接口如下：
add_image(tag, img, step, walltime=None)
参数	    格式	                    含义
tag	        string	        记录指标的标志，如train/loss，不能含有%
img	        numpy.ndarray	以ndarray格式表示的图片
step	    int	            记录的步数
walltime	int	            记录数据的时间戳，默认为当前时间戳
'''
'''
import numpy as np
from PIL import Image
from visualdl import LogWriter

def random_crop(img):
'''
    #获取图片的随机100X100分片
'''
    img = Image.open(img)
    #print(img)
    w, h = img.size
    print(w, h)
    random_w = np.random.randint(0, w - 100)
    random_h = np.random.randint(0, h - 100)
    r = img.crop((random_w, random_h, random_w + 300, random_h + 300))
    return np.asarray(r)

if __name__ == '__main__':
    #初始化一个记录器
    with LogWriter(logdir="./log/image_test/train") as writer:
        for step in range(20):
            #添加一个图片数据
            writer.add_image(tag="cat", img=random_crop("cat.jpg"), step=step)
'''


'''
High Dimensional （数据降维组件）组件的记录接口如下：
add_embeddings(tag, labels, hot_vectors, walltime=None)

参数	    格式	                    含义
tag	        string	                记录指标的标志，如default，不能含有%
labels	    numpy.array 或 list 	一维数组表示的标签，每个元素是一个string类型的字符串
hot_vectors	numpy.array or list 	与labels一一对应，每个元素可以看作是某个标签的特征
walltime	int	                    记录数据的时间戳，默认为当前时间戳
'''


from visualdl import LogWriter
import numpy as np

if __name__ == '__main__':
    hot_vectors = np.random.randn(300)
    hot_vectors = hot_vectors.reshape(100, 3)

    labels = 'label_'
    num = np.arange(300)

    labels = []
    for i in range(1, 301):
        lab = 'label_'
        if i < 10:
            lab = lab + str(0)
        labels.append(lab+str(i))
    #print(label)

    #初始化一个记录器
    with LogWriter(logdir="./log/high_dimensional_test/train") as writer:
        #将一组labels和对应的hot_vectors传入记录器进行记录
        writer.add_embeddings(tag='default',
                                    labels=labels,
                                    hot_vectors=hot_vectors)
