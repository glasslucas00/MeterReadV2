from  MeterClass import *
if __name__ =="__main__":  

    #多张图片，修改输入文件夹

    # imglist=glob.glob('input/*.jpg')  
    # for imgpath in  imglist: 
    #     A=MeterDetection(imgpath)
    #     A.Readvalue()
    #一张图片
    imgpath='images/1.jpg'
    A=MeterDetection(imgpath)
    readValue=A.Readvalue()
