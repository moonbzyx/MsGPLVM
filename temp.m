function [K, Knovar, Knophi, argExp] = rbfard2VardistPsi1ComputeMsPar(phi, rbfardKern, vardist, Z)
    % RBFARD2VARDISTPSI1COMPUTEMSPAR description.
    % INPUT: 
    % phi NxD;
    % Z:MXQ
    % VARGPLVM
    
    % variational means
    N  = size(vardist.means,1); % 100
    M = size(Z,1); % 50 
    % D = size(phi,1)/N;
    D=size(phi,2); % 30
    A = rbfardKern.inputScales; % alpha w [ ] : 1 x 8
             
    argExp = zeros(N,M); 
    normfactor = ones(N,1);
% test git_push
% from matlab to python

    for q=1:vardist.latentDimension
        S_q = vardist.covars(:,q);  
        normfactor = normfactor.*(A(q)*S_q + 1);
        Mu_q = vardist.means(:,q); 
        Z_q = Z(:,q)';
        distan = (repmat(Mu_q,[1 M]) - repmat(Z_q,[N 1])).^2;
        argExp = argExp + repmat(A(q)./(A(q)*S_q + 1), [1 M]).*distan;
    end
    
    normfactor = normfactor.^0.5;
    Knovar = repmat(1./normfactor,[1 M]).*exp(-0.5*argExp); 
    Knophi = rbfardKern.variance*Knovar; 
    Knovart = bsxfun(@times,repmat(reshape(phi,[N 1 D]),[1 M 1]),Knovar);
    Knovart = mat2cell(reshape(Knovart,N,M*D),N,M*ones(1,D));
    K = cellfun(@(x)(rbfardKern.variance*x),Knovart,'UniformOutput',0);
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
