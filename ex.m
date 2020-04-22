%% 导入数据

clear 
load('hw3.mat');
load('w_optimal.mat');
[row,column]=size(X);
Z=zeros(row,column);
[max_column,index_max]=max(X);
[min_column,index_min]=min(X);
for j=1:column
    Z(:,j)=(X(:,j)-min_column(j))/(max_column(j)-min_column(j));
end
I=ones(row,1);
Z_hat=[I Z];
X_hat=[I,X];
%% 近似最大特征值
ZTZ=Z_hat'*Z_hat;
ZTZ_norm=ZTZ;
for j=1:column+1
    ZTZ_norm(:,j)=ZTZ(:,j)/sum(ZTZ(:,j));
end
W_ZTZ=zeros(column+1,1);
for i=1:column+1
    W_ZTZ(i)=sum(ZTZ_norm(i,:));
end
W_ZTZ=W_ZTZ/sum(W_ZTZ);
lamada_max=sum(ZTZ*W_ZTZ);

XTX=X_hat'*X_hat;
XTX_norm=XTX;
for j=1:column+1
    XTX_norm(:,j)=XTX(:,j)/sum(XTX(:,j));
end
W_XTX=zeros(column+1,1);
for i=1:column+1
    W_XTX(i)=sum(XTX_norm(i,:));
end
W_XTX=W_XTX/sum(W_XTX);
lamada_max_x=sum(XTX*W_XTX);
L_x=2/row*lamada_max_x;


Z_hat=sparse(Z_hat);
ZTZ=sparse(ZTZ);

%% 闭式最优解
w_optimal=inv(Z_hat'*Z_hat)*Z_hat'*y;

%% 学习率
L=2/row*lamada_max;
alpha=1.8/L;
%% 梯度下降
u=zeros(column+1,1);
times=100000;
y_const=Z_hat'*y;
g_u_optimal=sum((y-Z_hat*u_optimal).^2)/row;
g_u=sum((y-Z_hat*u).^2)/row;
gradien_const=2/row*alpha;
gradien=2/row*(ZTZ*u-y_const);
%gradien_buffer=gradien;
g_buffer=zeros(times/50+1,1);
g_buffer(1)=g_u;
f_buffer=zeros(times/50+1,1);
u_w=[1 (max_column-min_column)]';
f_buffer(1)=sum((y-X_hat*(u./u_w)).^2)/row;
error=1e-2;

k=0;
% tic
% for i=1:times
%     if(abs(g_u_optimal-g_buffer(i))>error)
%          u=u-alpha*gradien;       
%          gradien=2/row*(ZTZ*u-y_const);
%          g_buffer(i+1)=sum((y-Z_hat*u).^2)/row;
%          f_buffer(i+1)=sum((y-X_hat*(u./u_w)).^2)/row;
% %         g_buffer(i)=g_u;
%     else
%         break
%     end
% %     if(mod(k,50)==0)
% %         g_w=sum((y-Z_hat*w).^2)/row;
% %     end
% end
% toc
tic
for i=1:times
         u=u-alpha*gradien;       
         gradien=2/row*(ZTZ*u-y_const);
    if(mod(i,50)==0)
         g_buffer(i/50+1)=sum((y-Z_hat*u).^2)/row;
         f_buffer(i/50+1)=sum((y-X_hat*(u./u_w)).^2)/row;
        if(abs(g_u_optimal-g_buffer(i/50+1))<error) 
        break;
        end
    end
end  
toc
g_buffer=g_buffer(1:i/50+1);
f_buffer=f_buffer(1:i/50+1);
plot(g_buffer);
figure
plot(f_buffer);

%% u与w关系
%w=u./(max_column-min_column);





