function [Result1,Result2]=VPCNN(matrix,Para)

link=Para.link;
np=Para.iterTimes;
alpha_L=Para.alpha_L;
alpha_Theta=Para.alpha_Theta ;
beta=Para.beta;
vL=Para.vL;
vTheta=Para.vTheta;
%=============================================================
[lenth,with,bands]=size(matrix);
F_NA=abs(matrix);

Y=zeros(3*lenth,3*with,bands);
L=zeros(3*lenth,3*with,bands);
Y0=zeros(lenth,with,bands);
Theta=zeros(3*lenth,3*with,bands);
% Compute the linking strength.
center=round(link/2);
W1=zeros(link,link);
R=zeros(bands,1);
for i=1:link
    for j=1:link
        if (i==center)&&(j==center)
            W1(i,j)=0;
        else
            W1(i,j)=1./sqrt((i-center).^2+(j-center).^2);
        end
    end
end
F_N=F_NA;
for n=1:np
    % padding avoid edges 
    for ii=1+lenth:2*lenth
        for jj=1+with:2*with
            F=F_N(ii-lenth,jj-with,:);
            F=F(:);                           % input F
            W=W1;
            YY=Y(ii-center+1: ii+center-1, jj-center+1: jj+center-1, :);   % input Y
            K=zeros(bands,1);
            for b1=1:link
                for b2=1:link
                    YT=YY(b1,b2,:);
                    YT=YT(:);
                    K=K+W(b1,b2).*YT;           % coefficient of input Y
                end
            end
            LL=L(ii,jj,:);
            LT=LL(:);
            L1=exp(-alpha_L)*LT+vL*K;                      % weight L
            L(ii,jj,:)=reshape(L1,1,1,bands);              % update L
            U=F.*(1+beta*L1);                              % output U
            TH=Theta(ii,jj,:);
            TH=TH(:);
            E=exp(-alpha_Theta)*TH+vTheta*R;               % feedback E
            R=abs(U-E);
            Y(ii,jj,:)=reshape(R,1,1,bands);               % update Y
            Theta(ii,jj,:)=reshape(E,1,1,bands);           % update E
        end
    end
    FNA=Y(lenth+1:2*lenth, with+1:2*with, :);             % output Y
    Y0=Y0+FNA;                                            % output Y0
end
Result1=Y0;
Result2=FNA;