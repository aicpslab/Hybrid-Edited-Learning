clear all
close all
clc
load('DataSet/MPC_Controller/MPC_DataGenerated.mat')
maximum_entropy=2000;
Structure=[12;4;2];
initial_rank=4;
tol = 2;
xs=xs'; %loading the data set
t=y';
% How many trajectories we want to obtain
[coeff,scoreeeTrain,~,~,explained,mu] = pca(xs);
idx=find(cumsum(explained)>90,1);      
%idx=2;
utest = (xs-mu)*coeff(:,1:idx);
bounderies = zeros(idx,2);

for i= 1:idx
    bounderies(i,1)=min(utest(:,i))-2;
    bounderies(i,2)=max(utest(:,i))+2;
end
lowerbound=bounderies(:,1);
upperbound= bounderies(:,2);
%init_interval{1}=[lowerbound',upperbound'];
init_interval{1}=bounderies;
Dimension=size(bounderies,1);

%% Data-driven Partitioning
P=partitions(init_interval,xs,t);
intervals=ME(P,tol,maximum_entropy,Dimension,mu',coeff(:,1:idx)');
P1=P;    

%% Train sub ELMs via Physics Regulated Method

 Phy_lay=ones(2,12);
% for i = 1:3
%     Phy_lay(2,3+(i-1)*4)=0;
% end
% figure
% partitions.intervalplot(intervals,'empty','black')

%Structure=[TaylorExpansionRank,OutputNeuronNumber]
%Structure=[3,10;2,2];

Inputgate=zeros(size(xs,2),3);
%Inputgate=ones(size(xs,2),3);

%WeightGateTemp=zeros(1819,4);
for i = 1:3
   Inputgate((i-1)*4+1:(i-1)*4+4,i)=ones(4,1);
  % Inputgate(1:4,i)=ones(4,1);
   WeightGateTemp(:,i)=PhyInput(Inputgate(:,i),Structure(2,1)); 
end
 
WeightGate=zeros(size(WeightGateTemp,1),1);
for i = 1:size(WeightGateTemp,1)
   WeightGate(i,:)=max(WeightGateTemp(i,:));
end
WeightGate=repmat(WeightGate,1,Structure(3,1))';
phtayELMs1=phtayHiddenELM.GeneratephtayHiddenELM(Structure,WeightGate,'poslin');
%% Input Weight optimized

%phtayELMs1.weight{1,1}=phtayELMs1.weight{1,1}.*WeightGate;
[~, inputSettings] = mapminmax(xs');
[~, outputSettings] = mapminmax(t');
%% Train sub-phytay ELMs
for j = 1 : size(intervals,2) 
     initial_rank=4;
     inputforall = xs; %loading the data set
     outputforall = t; %loading the data set
     [input,output] = P.Dataselect(inputforall,outputforall,intervals{j},Dimension,mu',coeff(:,1:idx)',size(inputforall,2));
     % IN=mapminmax('apply',input{1},inputSettings);
     % OUT=mapminmax('apply',output{1},outputSettings);
     %  phtayELM(j)=trainphtayELM(phtayELMs1,IN,OUT);
     IN=mapminmax('apply',input{1},inputSettings);
     OUT=mapminmax('apply',output{1},outputSettings);
     %phtayELM(j)=trainphtayHiddenELM(phtayELMs1,input{1},output{1});
     tic
     phtayELM(j)=trainphtayHiddenELM(phtayELMs1,IN,OUT);
     
     % while phtayELM(j).trainingError>0.1
     %        %   Weight_expansion
     %  initial_rank=initial_rank+1;
     %      for i = 1:3
     %           WeightGateTemp1(:,i)=PhyInput(Inputgate(:,i),initial_rank); 
     %      end
     %  WeightGate=zeros(size(WeightGateTemp1,1),1);
     %  for i = 1:size(WeightGateTemp1,1)
     %       WeightGate(i,:)=max(WeightGateTemp1(i,:));
     %  end    
     %         WeightGate=repmat(WeightGate,1,Structure(3,1))';
     %         phtayELMs=phtayHiddenELM.GeneratephtayHiddenELM([12;initial_rank;2],WeightGate,'poslin');
     %         phtayELM(j)=trainphtayHiddenELM(phtayELMs,input{1},output{1});
     % WeightGateTemp1=[];
     % end
     fprintf('training time is')
     toc
     %fprintf('training error is')
     %disp(phtayELM(j).trainingError)
end


%% Now starts the MPC process
load('DataSet/four obstacles MPC Test_Noa.mat');
%load("follower_carB_state.mat")
% Sample Number
xled = xOpt(1,:);
yled = xOpt(2,:);
vled = xOpt(3,:);
thetaled = xOpt(4,:);
% sampling time
TS = 0.2;
lr = 3;
lf = 3;
% generate leading car (carA) state data for testing purposes
nx = 101;
timestep = 100; 

% plot(xled(:,1:size(xled,2)),yled(:,1:size(xled,2)))
% title("speed vs position plot")

% obtain the state reference of carA
zref = [xled; yled; vled; thetaled];
N = size(xled,2)-1;
% define each single mpc computing horizon
P = 3;
% initialize the initial conditions for the following car (carB)
x0 = 0; %position
y0 = 0; %position
v0 = 5; %speed
theta0 = 0; %heading angle
t0 = 0; 
z0B = [x0; y0; v0; theta0]; %state
nzB = size(z0B,1); %row size of state
% initilize initial input of the carB
a0 = 0; % accelaration
deltaf0 = 0; % steering angle
u0B = [a0; deltaf0];
nuB = size(u0B,1);
% define the variable to be optimized
zB = z0B;
uB = u0B;
% tune the Q and R
QB = 12*eye(nzB);
% less penalty on the speed difference
QB(3,3) = 3; 
RB = [0 0; 0 0];
% save the states and input
usaveB = zeros(nuB,N);
zsaveB = zeros(nzB,N);
zsaveB(:,1) = z0B;
cons(:,1) = [-1.5;-60*pi/180];
cons(:,2) = [4;60*pi/180];
    
umin = min(t(:,1:2))';
umax = max(t(:,1:2))';
xmax = max(xs)';
xmin = min(xs)';
% (std(t))
for i=1:N-P
  %  tic

    % if i<=size(yled)-10
    %     sumy = sum(yled(i:i+10));
    % else 
    %     sumy = sum(yled(i+5:N));
    % end

    % threshold = 3;
    % if sumy>=threshold
    %     safe = [-0.03*(lr+lf);0*(lr+lf);0;0];
    % elseif sumy<=-threshold
    %     safe = [-0.03;0*(lr+lf);0;0];
    % else 
    %     safe= [0;0;0;0];
    % end

   % bar_zrefB = zref(:,i);
    objB = 0;
    %if i<=10
      %  usaveB(:,i) = zeros(nuB,1);
      %  zsaveB(:,i+1) = zeros(nzB,1);
   % else
  
        % for j =1:P
        %     %betaB(j) = atan((lr*tan(double(u(2,j))))/(lf+lr));
        %     if abs(zsa-veB(1,i)-zref(1,i-10))<=3*(lr+lf) || abs(zsaveB(2,i)-zref(2,i-10)<=3*(lr+lf))
        %         objB = objB + (zB(:,j)-(bar_zrefB+safe))'*QB*(zB(:,j)-(bar_zrefB+safe)) + uB(:,j)'*RB*uB(:,j);
        %     else
        %         objB = objB + (zB(:,j)-(bar_zrefB+safe))'*QB*(zB(:,j)-(bar_zrefB+safe)) + uB(:,j)'*RB*uB(:,j);
        %     end
        % 
        %     bar_zrefB(1) = bar_zrefB(1)+TS*bar_zrefB(3)*cos(bar_zrefB(4));
        %     bar_zrefB(2) = bar_zrefB(2)+TS*bar_zrefB(3)*sin(bar_zrefB(4));
        % end

       % constraintB = zB(:,1) == zsaveB(:,i);
   
        betaB = atan((lr*tan(uB(2,i)))/(lf+lr));
        %constraintB = [constraintB, zmin <= zB(:,j) <= zmax,...
         %   umin <= uB(:,j) <= umax,...
            zB(1,i+1) = zB(1,i)+TS*zB(3,i)*cos(zB(4,i)+betaB);
            zB(2,i+1) = zB(2,i)+TS*zB(3,i)*sin(zB(4,i)+betaB);
            zB(3,i+1) = zB(3,i)+TS*uB(1,i);
            zB(4,i+1) = zB(4,i)+TS*zB(3,i)*sin(betaB)/lr;
%% Transfer into relative coordinade 

%   for j = 1:1
 
    headxs=1;
      for j = 1:P
        head = (j-1)*4+1;
        % if zref(1,i)-zB(1,i)==0
        % xp(head-4,1)  = 0;
        % xp(head-3,1) = 0;
        % xp(head-2,1) = 5;
        % xp(head-1,1) = 0;
        % 
        % else
        xp(head:head+3,1)=zref(:,i+j-1)-zB(:,i);
        % xp(head-4,1) = sin(pi/2-atan((zref(2,i)-zB(2,i)/(zref(1,i)-zB(1,i))))-zref(1,i))*sqrt((zref(1,i)-zB(1,i))^2+(zref(2,i)-zB(2,i))^2);
        % xp(head-3,1) = cos(pi/2-atan((zref(2,i)-zB(2,i)/(zref(1,i)-zB(1,i))))-zref(1,i))*sqrt((zref(1,i)-zB(1,i))^2+(zref(2,i)-zB(2,i))^2);
        % xp(head-2,1) = zref(3,i)*cos(zref(4,i)-zB(4,i))-zB(3,i);
        % xp(head-1,1) = zref(3,i)*sin(zref(4,i)-zB(4,i));
        % end
      end


% 计算自车坐标系下的相对速度
%xp(3,1) =  cos_theta * delta_vx + sin_theta * delta_vy;  % 纵向相对速度
%xp(4,1)  = -sin_theta * delta_vx + cos_theta * delta_vy;  % 横向相对速度
 % end
segmentIndex = intervals;
inputspace1 = intervals;
flag=0;
for k = 1:size(segmentIndex,2)
     if(partitions.ifin(coeff(:,1:idx)'*(xp-mu'),segmentIndex{k},Dimension)==1)
         tic       
         % uB(:,i+1)= phtayHiddenELMpredict(phtayELM(k),xp);
         %        flag=1;
          INt=mapminmax('apply',xp,inputSettings);       
           OUTT = phtayHiddenELMpredict(phtayELM(k),INt);
          uB(:,i+1)=mapminmax('reverse',OUTT,outputSettings);
          toc
                %  INt=mapminmax('apply',xp,inputSettings);       
          %  OUTT = phtayELMpredict(phtayELM(k),INt);
          % uB(:,i+1)=mapminmax('apply',OUTT,outputSettings);
                flag=1;
     end
end
if(~flag)
uB(:,i+1)=uB(:,i);
end
if uB(1,i+1)<umin(1,1)
  uB(1,i+1)=umin(1,1);
end
if uB(1,i+1)>umax(1,1)
    uB(1,i+1)=umax(1,1);
end
if uB(2,i+1)<umin(2,1)
  uB(2,i+1)=umin(2,1);
end

if uB(2,i+1)>umax(2,1)
    uB(2,i+1)=umax(2,1);
end

if zB(3,i+1)<xmin(3,1)
  zB(3,i+1)=xmin(3,1);
end
if zB(3,i+1)>xmax(3,1)
   zB(3,i+1)=xmax(3,1);
end
if zB(4,i+1)<xmin(4,1)
  zB(4,i+1)=xmin(4,1);
end
if zB(4,i+1)>xmax(4,1)
   zB(4,i+1)=xmax(4,1);
end

    %bar_zrefB = zref(:,i-10);
    
  %  end
        % setup yalmip
        %options = sdpsettings('verbose',0);
        %sol = optimize(constraintB, objB, options); 
        % forward pass
%       toc
        usaveB(:,i) = uB(:,i+1);
        zsaveB(:,i+1) = zB(:,i+1);
end
    % define the safety difference based on obstacle side
figure
    plot(zB(1,:),zB(2,:),'-x')
    hold on 
    plot(zref(1,:),zref(2,:),'-o')
    xlabel('x_1 (m)')
    ylabel('x_2 (m)')
    legend('Vehicle','Reference')
    zBBPMPC=zB;
   % save('zBMPCBP.mat', 'zBBPMPC');
 
 %print(gcf, 'UneditedControl.png', '-dpng', '-r900')  
 figure
    plot(1:size(zB,2),zB(4,:))
    hold on
    plot(1:size(zref,2),zref(4,:))
 error=0;
 location=0;
 for i = 1:size(phtayELM,2)
     if phtayELM(i).trainingError>error
         error=phtayELM(i).trainingError;
         location = i;
     end
 end
 [input,~] = P1.Dataselect(inputforall,outputforall,intervals{location},Dimension,mu',coeff(:,1:idx)',size(inputforall,2));
disp(mse(zref(3:4,:),zB(3:4,:))) 
 fprintf('The WORST training error is')
 disp(error)
 fprintf('at location')
 disp(location)
 fprintf('where sample Num is')
 disp(size(input{1},2))
  %% Gerneral Error
Error=zeros(4,1);
Counter=0;
xs=xs';
for i = 1:size(xs,2)-1
    for k = 1:size(segmentIndex,2)
         if(partitions.ifin(coeff(:,1:idx)'*(xs(:,i)'-mu'),segmentIndex{k},Dimension)==1)
                    Control= phtayHiddenELMpredict(phtayELM(k),xs(:,i));
                    flag=1;
         Counter=Counter+1;
         end
        
    end
    if flag 
    betaB = atan((lr*tan(Control(2,1)))/(lf+lr));
        %constraintB = [constraintB, zmin <= zB(:,j) <= zmax,...
         %   umin <= uB(:,j) <= umax,...
         Traget(1,1) = xs(1,i)+TS*xs(3,i)*cos(xs(4,i)+betaB);
         Traget(2,1) = xs(2,i)+TS*xs(3,i)*sin(xs(4,i)+betaB);
         Traget(3,1) = xs(3,i)+TS*Control(1,1);
         Traget(4,1) = xs(4,i)+TS*xs(3,i)*sin(betaB)/lr;
         Error=Error+(xs(1:4,i+1)-Traget).^2;
    end
end
diva=std(xs');
for j=1:2
   NRMSE_Error(j,1)=sqrt(1/size(xs,1)*1/(diva(1,j)^2*Error(j,1)));
% NRMSE_Error(j,1)=sqrt(1/size(xs,1)*Error(j,1));
end
fprintf('NRMSE is')
disp(NRMSE_Error)


diva2=std(zref');
SimulationError=zeros(2,1);
for i = 1:2
    %SimulationError(i,1)=sqrt(1/(diva2(1,i)^2*mse(zB(i,1:end),zref(i,1:98))));
    SimulationError(i,1)=mse(zB(i,1:end),zref(i,1:98));
end
fprintf('NRMSE of the simulation Results')
disp(SimulationError)
%% Weight Expansion
function MXR=PhyInput(X,r)
        MXR = [];
N=size(X,1);
Q=size(X,2);
% b. Generate Hidden Nodes
for l = 1:Q
    mxr = X(:, l);
    tempmxr = X(:, l);
    mcounting = (1:N)';
    for m = 2:r
        tb_list = [];
        for z = 1:N
            ta = X(z, l) * tempmxr(mcounting(z):end);
            if z == 1
                tb = ta;
            else
                tb = [tb; ta];
            end
            mcounting(z) = size(tb, 1) - length(ta) + 1;
        end
        tempmxr = tb;
        mxr = [mxr; tb];
    end

    if isempty(MXR)
        MXR = mxr;
    else
        MXR = [MXR, mxr];
    end
end
end
