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