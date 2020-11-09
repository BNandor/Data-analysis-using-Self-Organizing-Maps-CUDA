somcount=100
digitsize = 28
digitdimension = digitsize * digitsize
for s=0:(somcount-1)
  
  filename = sprintf("%d.som",s)
  mapdim = dlmread(filename, ' ', [0,0,0,1]);
  Mosaic=[]
  alldata = dlmread(filename,' ', 1,0);
  border = (max(max(alldata)))
  for i=1:mapdim(1)
    MosaicRow=[]
    for j=1:mapdim(2)    
      data = dlmread(filename, ' ',[(i-1)*mapdim(2)+j,0,(i-1)*mapdim(2)+j,digitdimension-1]);    
      MosaicRow=horzcat(MosaicRow,reshape(data(1,:),digitsize,digitsize)',(border*ones(digitsize,1)));    
    endfor
    Mosaic=vertcat(Mosaic,MosaicRow,border*ones(1,digitsize*mapdim(2)+mapdim(2)));
  endfor  

  image = imresize(mat2gray(Mosaic),2,"linear");
  figure(1);  
##  imwrite(image,sprintf("/home/spaceman/Msc-I/ML/MestInt/OCR/soms/emnist/images/%d.jpg",s))
  imshow(image);
  pause(0.01)
endfor