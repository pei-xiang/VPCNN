clc;
clear;
close all;
addpath(genpath(pwd));
tic;

load abu-urban-2.mat;
mask=map;


% parameter
Para.iterTimes=5; 
Para.link=3;
Para.alpha_L=0.1;
Para.alpha_Theta=0.1;
Para.beta=0.1;
Para.vL=0.1;
Para.vTheta=0.1;
order=0.9;

datas=data;
datas=double(datas);
img=datas;
[m,n,b]=size(img);

%  edge Expansion
w=1;
rh=zeros(m,n,b);
Img=zeros(m+2*w,n+2*w,b);
Img(w+1:m+w,w+1:n+w,:)=img;
Img(:,1:w,:)=Img(:,2*w:-1:w+1,:);
Img(:,n+w+1:n+2*w,:)=Img(:,n+w:-1:n+1,:);
Img(1:w,:,:)=Img(2*w:-1:w+1,:,:);
Img(m+w+1:m+2*w,:,:)=Img(m+w:-1:m+1,:,:);

%  local correlation FrFT
for ki=1+w:m+w
    for kj=1+w:n+w
        T=Img(ki,kj,:);  T=T(:);
        block = Img(ki-w: ki+w, kj-w: kj+w, :);
        block(2,2,:)=NaN;
        block = reshape(block, 9, b);
        block(isnan(block(:, 1)), :) = []; 
        H = block';
        tep=zeros(b,8);
        val=zeros(1,8);
        tep(:,1)=(H(:,1)+H(:,4)+H(:,2)+H(:,3)+H(:,5))./5-T;
        tep(:,2)=(H(:,4)+H(:,6)+H(:,7)+H(:,5)+H(:,8))./5-T;
        tep(:,3)=(H(:,1)+H(:,4)+H(:,6)+H(:,2)+H(:,7))./5-T;
        tep(:,4)=(H(:,3)+H(:,5)+H(:,8)+H(:,2)+H(:,7))./5-T;
        tep(:,5)=(H(:,1)+H(:,4)+H(:,6)+H(:,2)+H(:,3))./5-T;
        tep(:,6)=(H(:,1)+H(:,4)+H(:,6)+H(:,7)+H(:,8))./5-T;
        tep(:,7)=(H(:,3)+H(:,5)+H(:,8)+H(:,1)+H(:,2))./5-T;
        tep(:,8)=(H(:,3)+H(:,5)+H(:,8)+H(:,6)+H(:,7))./5-T;
        for t=1:8
            val(t)=norm(tep(:,t));
        end
        [value,index]=min(val);
        cha=T+tep(:,index);
        rh(ki-w,kj-w,:)=reshape(cha',1,1,b);
    end
end
img1 = rh./max(rh(:));
im1 = zeros(m, n, b);
for i = 1:m
    for j = 1:n
        % amptitude
        im1(i,j,:) = Disfrft(squeeze(img1(i,j,:)),order);
    end
end
tet1 = im1;

% vector pulse coupled neural network
[R1,R2]=VPCNN(tet1,Para);


f1=sum((R1).^2,3);
fvpcnn=mat2gray(f1);
figure;
imshow(fvpcnn);
figure;imagesc(mat2gray(fvpcnn));axis image;
toc;

%% ROC curve
disp('Running ROC...');
MM=m*n;
r4=reshape(fvpcnn,1,MM);
mask = reshape(mask, 1, MM);
anomaly_map = logical(double(mask)>=1);
normal_map = logical(double(mask)==0);
r_max = max(r4(:));
taus = linspace(0, r_max, 20000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r4 > tau);
  PF1(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD1(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area4 =  sum((PF1(1:end-1)-PF1(2:end)).*(PD1(2:end)+PD1(1:end-1))/2);
figure;
plot(PF1, PD1, 'k-', 'LineWidth', 2);  grid on
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('FVPCNN');
axis([0 1 0 1]);