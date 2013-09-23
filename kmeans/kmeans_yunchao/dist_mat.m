function D=dist_mat(P1, P2)
%
% Euclidian distances between vectors
P1 = double(P1);
P2 = double(P2);

X1=repmat(sum(P1.^2,2),[1 size(P2,1)]);
X2=repmat(sum(P2.^2,2),[1 size(P1,1)]);
R=P1*P2';
D=sqrt(X1+X2'-2*R);
