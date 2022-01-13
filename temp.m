% from matlab to python

%% the parameters setting
model.alpha = options.alpha *rand(1,model.L);
model.beta = options.beta *rand(1,model.D);
v=zeros(model.D,model.L);
for l=1:model.L
    v(:,l) = gamrnd(model.alpha(l),1,[model.D,1]);
end
v=v./repmat(sum(v,2),[1,model.L]);
% z=mnrnd(
% z=zeros(model.N,model.D,model.L);
% for d=1:model.D
%     ind = crossvalind('Kfold',model.N,model.L)';
%     for l=1:model.L
%         z((ind==l),d,l)=1;
%     end
% end
z = rand(model.N,model.D,model.L);

% % the parameters of the kernel
% model.sigma=exp(2*log(rand(L,1)));
% w=exp(log(rand(L,model.Q)));% can w be negative? no!
% model.w=w;

% how to generate a different kernel for the same data set.怎么初始化呢
inputScales = 5./(((max(X)-min(X))).^2);
varss  = var(model.m(:));
for l=1:model.L
    if isstruct(options.kern)
        model.kern{l} = options.kern;
    else
        model.kern{l} = kernCreateMs(model.X, options.kern,l);
        %         rand('seed', l);
        %         model.kern{l}.variance = l*l*varss/model.L/model.L;
        %         model.kern{l}.inputScales = rand(1,model.kern{l}.inputDimension);
    end
end