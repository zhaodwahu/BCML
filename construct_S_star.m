 
function  [S0] =  construct_S_star(dzx,S,theta,lambda1, k, n, m) 
%         S0 = zeros(n,m);
%         distXF = distX+lambda2*L2_distance_1(F',G');  % ||x_i - x_j ||^2
        
%         distX1= lambda1*Kzx';
        distXF= theta*S+lambda1*dzx;
        [~, idx] = sort(distXF,2,'descend');  % %dim==2，按行排序
        S0 = zeros(n,m);
        for i=1:n
            idxa0 = idx(i,2:k+1);
            dxi = distXF(i,idxa0);
%             di0 = distX1(i,2:k+2);
%             di1 = S(i,2:k+2);
%             rr(i) = 0.5*(k*di0(k+1)-sum(di0(1:k)));
%             r_up=sum(di0(1:k))-k*lambda1*di0(k+1);
%             r_down=theta*k*di1(k+1);
%             rr(i)=r_up./r_down;
            ad = (dxi)/(theta);
            S0(i,idxa0) = EProjSimplex_new(ad);
        end
%         r = mean(rr);
        
%         S{v,1} = S0;
end