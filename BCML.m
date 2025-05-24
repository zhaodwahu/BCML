
function [model_LVSL] = BCML( X_set, Y, optmParameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% optimization parameters
    lambda1          = optmParameter.lambda1;
    lambda2          = optmParameter.lambda2;
    lambda3          = optmParameter.lambda3;
    lambda4          = optmParameter.lambda4;
    lambda_inf       = optmParameter.lambda_inf;
    ratio            = optmParameter.ratio;

    maxIter          = optmParameter.maxIter;
    kernel_para      = optmParameter.kernel_para;
    kernel_type      = optmParameter.kernel_type;

   %% Initialization
    k=10;
    k1=5;
    num_views=length(X_set);
    
    [n,q]=size(Y);
    m=ceil(n*ratio);
    alpha = ones(num_views,1)/num_views;

    [Ktr,Kzx,Z,dxz]=Initializationpara(X_set,kernel_para,kernel_type,n,m);

    S=zeros(n,m);
     for ii=1:num_views
          T{ii,1}=pinv(Kzx{ii,1}'*Kzx{ii,1}+Ktr{ii,1}'*Ktr{ii,1}+lambda1*Ktr{ii,1});
          S_hat{ii,1} = ConstructA_NP(X_set{ii,1}', Z{ii,1}',k);
          S = S+alpha(ii)*S_hat{ii,1};
     end
   
    C=eye(q,q);
%% updating variables...
    iterVal = zeros(1,maxIter);
    for iter=1:maxIter
           if optmParameter.outputthetaQ
               fprintf('- iter - %d/%d\n', iter, maxIter);
           end

           Y_bar = Y;
           D = diag(sum(S));

           XW = zeros(n,q);
           ZW = zeros(m,q);
            %% Update F & G
                R=pinv(lambda3*D+(1+lambda_inf)*speye(m)); 
                Q=1/(1+lambda_inf+lambda3)*(lambda_inf*Y_bar*C+XW+lambda3*S*R*ZW);
                M=(lambda3^2/(1+lambda_inf+lambda3))*S*R;
    
                F=(speye(n)+M*pinv(speye(m)-S'*M)*S')*Q;
                % Updata matrix G
                G=R*(lambda3*S'*F+ZW);

                C_old=updateC(C,Y_bar,F,lambda4,k1);
                C = C_old; 

           for v=1:num_views      
                 %% W
                 W{v,1}=T{v,1}*(Ktr{v,1}'*F+Kzx{v,1}'*G);
           
            %% S*v

            [S_star{v,1}] =  construct_S_star(dxz{v,1},S,alpha(v,1),lambda3, k, n, m); 

            XW=XW+alpha(v)*Ktr{v,1}*W{v,1};
            ZW=ZW+alpha(v)*Kzx{v,1}*W{v,1};
            prediction_Loss(v,1)=trace((Ktr{v,1}*W{v,1}-F)'*(Ktr{v,1}*W{v,1}-F))+lambda1*trace(Ktr{v,1}*W{v,1}*W{v,1}')...
                   +norm(S_star{v,1}-S,'fro')^2;
           end

           [S] =  construct_S1(S_star,F,G,alpha,lambda3, k, n, m) ;
             
           %% theta
           if optmParameter.updateTheta == 1
               alpha  =   construct_S_ln(prediction_Loss,lambda2, num_views);
           end

           if optmParameter.outputthetaQ == 1
               fprintf(' - prediction loss: ');
               for mm=1:num_views
                    fprintf('%e, ', prediction_Loss(mm));
               end
               fprintf('\n - theta: ');
               for mm=1:num_views
                    fprintf('%.3f, ', alpha(mm));
               end
               fprintf('\n');
           end

         diff = ((alpha)')*prediction_Loss + lambda2*alpha'*log(alpha)+lambda_inf*trace((F-Y_bar*C)'*(F-Y_bar*C));
         iterVal(iter) = diff;

        if abs(diff)<1e-5
            break
        end
        iter=iter+1;
    end 
            %% return values
            model_LVSL.W = W;
            model_LVSL.K = Ktr;
            model_LVSL.theta = alpha;
            model_LVSL.kernel_para = kernel_para;
            model_LVSL.kernel_type = kernel_type;
            model_LVSL.loss=iterVal;

end
function C=updateC(C_t,Ytrain,F,lambda1,k1)

%  [~, idx] = sort(C_t,2,'descend');
         kdtree = KDTreeSearcher(F');
        [neighbor,~] = knnsearch(kdtree,F','k',k1+1);
        neighbor = neighbor(:,2:k1+1);
 q=size(C_t,2);
 lr=1e-4;
F1=Ytrain'*Ytrain*C_t-Ytrain'*F;
F2=zeros(q);
for i=1:q
%     idxa0 = idx(i,2:k1+1);
    k0 = sum(exp(C_t(i,:))) - exp(C_t(i,i))+eps;
    for j=i:q
        if i~=j
            if ismember(j,neighbor(i,:))
               F2(i,j) = -1+exp(C_t(i,j))/k0;
            else 
               F2(i,j) = exp(C_t(i,j))/k0; 
            end
        end
    end
    
end

g=F1+0.5*lambda1*F2;
C=C_t-lr*g;

end

function [Ktr,Kzx,Z,dxz]=Initializationpara(X,kernel_para,kernel_type,n,m)
    
    for i=1:length(X)
       [~,Z{i,1},~]=kmeans(X{i,1},m);
        Ktr{i,1} = kernelmatrix(kernel_type,X{i,1}',X{i,1}',kernel_para);
        Kz{i,1}  = kernelmatrix(kernel_type,Z{i,1}',Z{i,1}',kernel_para);
        Kzx{i,1} = kernelmatrix(kernel_type,Z{i,1}',X{i,1}',kernel_para);
            for iii=1:n
                for jjj=1:m
                    dxz{i,1}=Ktr{i,1}(iii,iii)-2*Kzx{i,1}(jjj,iii)+Kz{i,1}(jjj,jjj);
                end
            end
    end

end

