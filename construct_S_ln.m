function  S =  construct_S_ln(dis,lambda, n)
S = zeros(n,1);
a = zeros(n,1);
for j = 1:n
    a(j,:) = sum(exp(-dis(j,:)./lambda));    
end
S = exp(-dis./lambda)/sum(a);
%     for k = 1:h
    
%     end
%     S(isnan(S))=0;
end