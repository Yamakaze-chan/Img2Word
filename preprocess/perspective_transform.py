from skimage.filters import threshold_local
import numpy as np
import cv2 as cv

img=cv.imread(r"test_img\26.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)
dilation = cv.dilate(gray,kernel,iterations =5)
blur =cv.GaussianBlur(dilation,(3,3),0)
blur= cv.erode(blur,kernel,iterations =5)
edge=cv.Canny(blur,100,200)

t=300;j=0
while(j<8 and t>0):     
    try:linesP=cv.HoughLines(edge,1,np.pi/180,t);j=linesP.shape[0]
    except:j=0
    t=t-10
lines=linesP.reshape(linesP.shape[0],2)
t=0;c=0;lu=[]
for l in lines:
    c=c+1;rho,theta=l
    for lt in lines[c:]:
        t=0
        if(lt[0]!=l[0]):
            rhot,thetat=lt;k=abs(lt-l)<[50,0.5] 
            if(k[0] and k[1]):
                t=-1;break                
    lu.append(l)
lr=np.asarray(lu[:4]);j=np.reshape(lr,[lr.shape[0],1,2])

def l_inter(line1, line2):
    r1, t1 = line1;r2,t2 = line2
    A= np.array([[np.cos(t1),np.sin(t1)],[np.cos(t2),np.sin(t2)]])
    b= np.array([[r1],[r2]]);x0,y0=(0,0)
    if(abs(t1-t2)>1.3):
        return [[np.round(np.linalg.solve(A, b))]]
def points_inter(lines):
    intersections = []
    for i, g in enumerate(lines[:-1]):
        for g2 in lines[i+1:]:
            for line1 in g:
                for line2 in g2:
                    if(l_inter(line1, line2)):
                        intersections.append(l_inter(line1, line2)) 
    return intersections
p=np.asarray(points_inter(j)).reshape(4,2)

r= np.zeros((4,2), dtype="float32")
s = np.sum(p, axis=1);r[0] = p[np.argmin(s)];r[2] = p[np.argmax(s)]
d = np.diff(p, axis=1);r[1] = p[np.argmin(d)];r[3] = p[np.argmax(d)]
(tl, tr, br, bl) =r
wA = np.sqrt((tl[0]-tr[0])**2 + (tl[1]-tr[1])**2 )
wB = np.sqrt((bl[0]-br[0])**2 + (bl[1]-br[1])**2 )
maxW = max(int(wA), int(wB))
hA = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2 )
hB = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2 )
maxH = max(int(hA), int(hB))
ds= np.array([[0,0],[maxW-1, 0],[maxW-1, maxH-1],[0, maxH-1]], dtype="float32")
transformMatrix = cv.getPerspectiveTransform(r,ds)
scan = cv.warpPerspective(gray, transformMatrix, (maxW, maxH))

cv.imshow("scan" , scan)
cv.imshow(0)