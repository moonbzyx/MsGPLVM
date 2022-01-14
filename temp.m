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

% reformulate the model.varZ.phi into the cell form.
phil = mat2cell(reshape(model.varZ.phi,[model.N*model.D,model.L]),model.N*model.D,ones(1,model.L));
phidd = cellfun(@(x)(reshape(x,model.N,model.D)),phil,'UniformOutput',0);
t3=cputime;

model.Stats.Psi0 = 
cellfun(@(phi,kernl)
    (cellfun(@(phi)(rbfard2VardistPsi0ComputeMs(phi,kernl,model.varX)),
        mat2cell(phi,model.N,ones(1,model.D)),'UniformOutput',0)),
    phidd,model.kern,'UniformOutput',0);


[model.Stats.Psi1,model.Stats.P1Knovar,model.Stats.P1Knophi] = cellfun(@(phi,kernl,xul)(rbfard2VardistPsi1ComputeMsPar(phi,kernl, model.varX, xul)),phidd,model.kern,Xu,'UniformOutput',0);
[model.Stats.Psi2,model.Stats.P2outKern, model.Stats.P2sumKern, model.Stats.P2Kgvar]= cellfun(@(phi,kernl,xul)(rbfard2VardistPsi2ComputeMsPar(phi,kernl, model.varX, xul)),phidd,model.kern,Xu,'UniformOutput',0);
% disp(['Psi Compute Time: ' num2str((cputime-t3))]);