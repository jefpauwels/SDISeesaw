% Seesaw script to compute bounds on Pguess under different SDI assumptions
% Requires YALMIP and MOSEK

% Seesaw function:
% Choose assumption
% = 1 (info) parameters = alpha (guess prob)
% = 2 (vac or almost dim) parameters = [eps targetdim]
% = 3 (distrust) parameters = {eps rho_target(:,:,x)}
% = 4 (dim assumption --> get ideal strategy) parameters = d

% Choose your guessing game
nY = 1;
nX = 3;
nB = nX;

c = zeros(nB,nX,nY);
 
 for x = 1:nX;for b = 1:nB;
         c(b,x,1) = [x==b]/nX;
 end;end

% almost dimension

targetdim = 3;
eps = 0.1;

universalBound = SDItoGuess(targetdim/nX,eps) %% Universal bounds, Eq. (7) of the paper "Information capacity of quantum communication under natural physical assumptions"

SeesawBound = UnifiedSeesaw(c,2,[eps,targetdim]) %% Heuristic lower bound

% distrust
d = 3;
eps = 0.1;

for x = 1:nX
rho_target(:,:,x) = RandomDensityMatrix(d,1,1)
end
alpa0_target = infoContent(rho_target)

universalBound=SDItoGuess(alpa0_target,eps) %% Universal bounds, Eq. (7) of the paper "Information capacity of quantum communication under natural physical assumptions"

SeesawBound = UnifiedSeesaw(c,3,{eps,rho_target}) %% Heuristic lower bound


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function eps = SDIInfo(alpha,alpha0,nX)
if alpha >1/nX
eps = alpha + alpha0 - 2 * alpha * alpha0 - 2 * sqrt(alpha * alpha0 - alpha^2 * alpha0 - alpha * alpha0^2 + alpha^2 * alpha0^2);
else 
    disp('info too low')
end
end

function alpha = SDItoGuess(alpha0,eps)
alpha= alpha0 + (1 - 2 * alpha0) * eps + 2 * sqrt(alpha0 * (1 - alpha0)) * sqrt(eps * (1 - eps));
end


%% Seesaws different assumptions

function [W rho] = UnifiedSeesaw(c,assumption,parameters)

[nB,nX,nY] = size(c);
if assumption ~=4 
D = 2*nX;
else % dimension bound
    D=parameters; %d
end
prec = 1/1e6;
diff = 1;
W1 = 0;

% get assumption-specific parameters
if assumption == 1 % info
    alpha = parameters;
    %D = 2*nX;
elseif assumption == 2 % almost dim or vacuum
    targetd = parameters(2);
    eps = parameters(1);
    Pi = zeros(D,D);
    Pi(1:targetd,1:targetd) = eye(targetd);
elseif assumption ==3 % distrust
    eps = parameters{1};
    rho_target_sub = parameters{2};
    d = size(rho_target_sub,1);
    for x = 1:nX
        rho_target(:,:,x) = zeros(D,D);
        rho_target(1:d,1:d,x) = rho_target_sub(:,:,x);
    end
end

% Init measurement
M = InitMeas(nB,nY,D);
count = 0;
while diff > prec & count <100;
    count = count +1;

W = 0;
rho = sdpvar(D,D,nX);

for x = 1:nX
    for y = 1:nY
        for b = 1:nB
            W = W + c(b,x,y)*real(trace(rho(:,:,x)*M(:,:,b,y)));
        end
    end
end

F = [];
if assumption == 1
sigma = sdpvar(D,D);
F = [F;sigma>=0;trace(sigma)<=alpha];
end
for x = 1:nX
    F = [F;trace(rho(:,:,x))==1]; % rho has trace 1
    F = [F;rho(:,:,x)>=0]; % rho is positive
    if assumption ==1
        F = [F;rho(:,:,x)/nX<=sigma];
    elseif assumption ==2
        F = [F;trace(Pi*rho(:,:,x))>=1-eps];
    elseif assumption ==3
        F = [F;trace(rho_target(:,:,x)*rho(:,:,x))>=1-eps];
    end
end


ops=sdpsettings('solver','mosek', 'verbose',0,'cachesolvers', 1);
solvesdp(F,-W,ops);
rho = double(rho);


[M,W] = OptMeas(c,rho);

diff = abs(W-W1);
W1 = W;
    
end


end



%% helpers

function M = InitMeas(nB,nY,D)
    
for y = 1:nY
    Meas = RandomPOVM(D,nB,1);
    for b = 1:nB
        M(:,:,b,y) = Meas{b};
    end
end

end

function [M,W] = OptMeas(c,rho)
D = size(rho,1);
[nB,nX,nY] = size(c);

W = 0;
M = sdpvar(D,D,nB,nY);

for x = 1:nX
    for y = 1:nY
        for b = 1:nB
            W = W + c(b,x,y)*real(trace(rho(:,:,x)*M(:,:,b,y)));
        end
    end
end

F = [];
for y = 1:nY
    s = 0;
    for b = 1:nB
        F = [F;M(:,:,b,y)>=0];
        s = s + M(:,:,b,y);
    end
    F = [F;s==eye(D)];
end

ops=sdpsettings('solver','mosek','verbose',0, 'cachesolvers', 1);
solvesdp(F,-W,ops);
M = double(M);
W = double(W);

end


function alpha0 = infoContent(rho)
d = size(rho,1);
nX = size(rho,3);

c = zeros(nX,nX,1);
for x = 1:nX
    c(x,x,1) = 1/nX;
end

[M alpha0] = OptMeas(c,rho);


end
