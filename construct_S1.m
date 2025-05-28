function  [S0] =  construct_S122(S_star,F,G,theta,lambda2, k, n, m) 
%         S0 = zeros(n,m);
        sumX=0;
        for v=1:length(S_star)
            sumX=sumX+theta(v)*S_star{v,1};
        end

%         distXF = r*sumX+lambda2*L2_distance_1(F',G');  % ||x_i - x_j ||^2
       %  distXF = sumX+lambda2*F*G';
          distXF = sumX+lambda2*L2_distance_1(F',G');          
        [~, idx] = sort(distXF,2,'descend');  % %dim==2，按行排序
        S0 = zeros(n,m);
        for i=1:n
            idxa0 = idx(i,2:k+1);
            dxi = distXF(i,idxa0);
%             di0 = distXF(i,2:k+2);
%             rr(i) = 0.5*(k*di0(k+1)-sum(di0(1:k)));
%             ad = (dxi)/(r);

            S0(i,idxa0) = EProjSimplex_new(dxi);
        end
%         r = mean(rr);
        
%         S{v,1} = S0;
end