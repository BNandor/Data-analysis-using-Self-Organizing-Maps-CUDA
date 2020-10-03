mapdim = dlmread('som.txt', ' ', [0,0,0,1]);
Mosaic=[]
alldata = dlmread('som.txt',' ', 1,0);
border = (max(max(alldata)))
for i=1:mapdim(1)
  MosaicRow=[]
  for j=1:mapdim(2)    
    data = dlmread('som.txt', ' ',[(i-1)*mapdim(2)+j,0,(i-1)*mapdim(2)+j,63]);    
    MosaicRow=horzcat(MosaicRow,reshape(data(1,:),8,8)',(border*ones(8,1)));    
  endfor
  Mosaic=vertcat(Mosaic,MosaicRow,border*ones(1,8*mapdim(2)+mapdim(2)));
endfor  

image = mat2gray(Mosaic);
figure(1);  
imshow(image)