function plot_result(X3D_ref,X3D_DL,X3D_rec,mask_3D)

band_set=[18 8 2];

figure()
subplot(1,4,1)
FalseColorf=X3D_ref(:,:,band_set);
RGBmax= max(X3D_ref(:));
RGBmin= min(X3D_ref(:));
FalseColorf= (FalseColorf-RGBmin)/(RGBmax-RGBmin);
ref=imadjust(FalseColorf,stretchlim(FalseColorf),[]);
imshow(ref);title('Reference');

subplot(1,4,2)
FalseColorf=X3D_ref(:,:,band_set);
RGBmax= max(X3D_ref(:));
RGBmin= min(X3D_ref(:));
FalseColorf= (FalseColorf-RGBmin)/(RGBmax-RGBmin);
xf=imadjust(FalseColorf,stretchlim(FalseColorf),[]);
cor=xf.*mask_3D(:,:,band_set(1)).*mask_3D(:,:,band_set(2)).*mask_3D(:,:,band_set(3));
imshow(cor);title('Corruption');

subplot(1,4,3)
FalseColorf=X3D_DL(:,:,band_set);
RGBmax= max(X3D_DL(:));
RGBmin= min(X3D_DL(:));
FalseColorf= (FalseColorf-RGBmin)/(RGBmax-RGBmin);
sDL=imadjust(FalseColorf,stretchlim(FalseColorf),[]);
imshow(sDL);title('sdADAM');

subplot(1,4,4)
FalseColorf=X3D_rec(:,:,band_set);
RGBmax= max(X3D_rec(:));
RGBmin= min(X3D_rec(:));
FalseColorf= (FalseColorf-RGBmin)/(RGBmax-RGBmin);
result=imadjust(FalseColorf,stretchlim(FalseColorf),[]);
imshow(result);title('ADMM-ADAM');