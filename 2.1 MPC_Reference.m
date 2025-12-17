%% put everything together
close all;
clear;
load('four obstacles MPC Test_Noa.mat');
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

 plot(xled(:,1:size(xled,2)),yled(:,1:size(xled,2)))
% title("speed vs position plot")

% obtain the state reference of carA
zref = [xled; yled; vled; thetaled];

%% mpc starts here
% define the total mpc iterations
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
zB = sdpvar(nzB,N+1);
assign(zB(:,1), z0B);
uB = sdpvar(nuB, N);
assign(uB(:,1), u0B);
% tune the Q and R
QB = 12*eye(nzB);
% less penalty on the speed difference
QB(3,3) = 3; 
RB = [0 0; 0 0];
% save the states and input
usaveB = zeros(nuB,N);
zsaveB = zeros(nzB,N);
zsaveB(:,1) = z0B;


% initialize cost function and constraintB
% define the state constraintBs
zmin = [-10;-100;0;-pi];
zmax = [1000;100;30;pi];
umin = [-1.5;-60*pi/180];
umax = [4;60*pi/180];
    

for i=1:N
    % initialize cost function and constraintB
    % set the reference as the carA's state at time i
    
    % define the safety difference based on obstacle side
    tic
    if i<=size(yled)-10
        sumy = sum(yled(i:i+10));
    else 
        sumy = sum(yled(i+5:N));
    end
    
    threshold = 3;
    if sumy>=threshold
        safe = [-0.03*(lr+lf);0*(lr+lf);0;0];
    elseif sumy<=-threshold
        safe = [-0.03;0*(lr+lf);0;0];
    else 
        safe= [0;0;0;0];
    end

    bar_zrefB = zref(:,i);
    objB = 0;
    if i<=10
        usaveB(:,i) = zeros(nuB,1);
        zsaveB(:,i+1) = zeros(nzB,1);
    else
        bar_zrefB = zref(:,i-10);
        for j =1:P
            %betaB(j) = atan((lr*tan(double(u(2,j))))/(lf+lr));
            if abs(zsaveB(1,i)-zref(1,i-10))<=3*(lr+lf) || abs(zsaveB(2,i)-zref(2,i-10)<=3*(lr+lf))
                objB = objB + (zB(:,j)-(bar_zrefB+safe))'*QB*(zB(:,j)-(bar_zrefB+safe)) + uB(:,j)'*RB*uB(:,j);
            else
                objB = objB + (zB(:,j)-(bar_zrefB+safe))'*QB*(zB(:,j)-(bar_zrefB+safe)) + uB(:,j)'*RB*uB(:,j);
            end

            bar_zrefB(1) = bar_zrefB(1)+TS*bar_zrefB(3)*cos(bar_zrefB(4));
            bar_zrefB(2) = bar_zrefB(2)+TS*bar_zrefB(3)*sin(bar_zrefB(4));
        end
        constraintB = zB(:,1) == zsaveB(:,i);
        for j = 1:P
            betaB(j) = atan((lr*tan(uB(2,j)))/(lf+lr));
            constraintB = [constraintB, zmin <= zB(:,j) <= zmax,...
                umin <= uB(:,j) <= umax,...
                zB(1,j+1) == zB(1,j)+TS*zB(3,j)*cos(zB(4,j)+betaB(j)),...
                zB(2,j+1) == zB(2,j)+TS*zB(3,j)*sin(zB(4,j)+betaB(j)),...
                zB(3,j+1) == zB(3,j)+TS*uB(1,j),...
                zB(4,j+1) == zB(4,j)+TS*zB(3,j)*sin(betaB(j))/lr];
        end
    

        % setup yalmip
        options = sdpsettings('verbose',0);
        sol = optimize(constraintB, objB, options); 
        % forward pass
       toc
        usaveB(:,i) = uB(:,1);
        zsaveB(:,i+1) = zB(:,2);
    end
    
    % define the safety difference based on obstacle side
    if i<=size(yled)-10
        sumy = sum(yled(i:i+10));
    else 
        sumy = sum(yled(i+5:N));
    end
    
    threshold = 3;
    if sumy>=threshold
        safe = [-0.03*(lr+lf);0*(lr+lf);0;0];
    elseif sumy<=-threshold
        safe = [-0.03;0*(lr+lf);0;0];
    else 
        safe= [0;0;0;0];
    end
    

end

time = linspace(0,timestep,nx-1);
figure;
plot(time,usaveB(1,:))
title("acceleration input of carB")
figure;
plot(xled(1,:), yled(1,:),'-o')
hold on;
plot(zsaveB(1,:),zsaveB(2,:), '-x')
xlabel('x_1')
ylabel('x_2')
legend("Reference","Vehicle");
print(gcf, 'MPCControl.png', '-dpng', '-r900')  
%title("carA's vs carB's positions")
hold off;

%% Animation

L = lr + lf;
l = 0.738;
nFrames = timestep;
F(nFrames) = struct('cdata', [], 'colormap', []);

figure;
hold on;


plot(xled, yled, 'o', 'LineStyle', '--', 'Color', 'r');

% B车
cxB = zsaveB(1, 1);
cyB = zsaveB(2, 1);
thetaB = zsaveB(4, 1);
LhB = line([cxB - L / 2 * cos(thetaB), cxB + L / 2 * cos(thetaB)], ...
    [cyB - L / 2 * sin(thetaB), cyB + L / 2 * sin(thetaB)], 'Color', 'blue');
LbB = line([cxB + L / 2 * cos(thetaB) - l / 2 * cos(usaveB(2, 1)), cxB + L / 2 * cos(thetaB) + l / 2 * cos(usaveB(2, 1))], ...
    [cyB + L / 2 * sin(thetaB) - l / 2 * sin(usaveB(2, 1)), cyB + L / 2 * sin(thetaB) + l / 2 * sin(usaveB(2, 1))], 'Color', 'blue');

disp('Creating and recording frames...')

for j = 1:nFrames
    % 更新B车的位置和角度
    cxB = zsaveB(1, j);
    cyB = zsaveB(2, j);
    thetaB = zsaveB(4, j);
    set(LhB, 'xdata', [cxB - L / 2 * cos(thetaB), cxB + L / 2 * cos(thetaB)], ...
        'ydata', [cyB - L / 2 * sin(thetaB), cyB + L / 2 * sin(thetaB)]);
    set(LbB, 'xdata', [cxB + L / 2 * cos(thetaB) - l / 2 * cos(usaveB(2, j)), cxB + L / 2 * cos(thetaB) + l / 2 * cos(usaveB(2, j))], ...
        'ydata', [cyB + L / 2 * sin(thetaB) - l / 2 * sin(usaveB(2, j)), cyB + L / 2 * sin(thetaB) + l / 2 * sin(usaveB(2, j))]);

    % 固定轴范围
    axis([0, 50, -5, 5]);
    axis square;

    drawnow;

    F(j) = getframe(gcf);
    pause(TS);  
end

disp('Playing movie...')
axis([0, 50, -5, 5]); 
axis square;
Fps = 5;
nPlay = 1;
movie(F, nPlay, Fps);


%% Animation
fig = figure;
hold on;

% 画底面
[X_bottom, Y_bottom] = meshgrid(0:1:70, -7:1:7);
Z_bottom = 0 .* X_bottom;
s_bottom = surf(X_bottom, Y_bottom, Z_bottom, 'FaceAlpha', 0.5);
s_bottom.EdgeColor = 'none';
s_bottom.FaceColor = 'black';

% 画侧面
[X_side, Z_side] = meshgrid(0:1:70, 0:0.5:6);
Y_side = 0 .* X_side + 7;
s_side = surf(X_side, Y_side, Z_side, 'FaceAlpha', 0.3);
s_side.EdgeColor = 'none';
s_side.FaceColor = 'green';

% 画分隔线
l1 = line([0, 70], [-2, -2]);
l2 = line([0, 70], [2, 2]);
l1.Color = [1, 1, 1];
l2.Color = [1, 1, 1];
% 
% % 画障碍物
% obstacles = [
%     10, 0, 1;
%     25, 1, 1.5;
%     40, 0, 1;
%     55, -1, 1.5
% ];
% 
% for i = 1:size(obstacles, 1)
%     [X_obs, Y_obs] = meshgrid(obstacles(i, 1)-obstacles(i, 3):0.5:obstacles(i, 1)+obstacles(i, 3), ...
%                             obstacles(i, 2)-obstacles(i, 3):0.5:obstacles(i, 2)+obstacles(i, 3));
%     Z_obs = 0 .* X_obs;
%     s_obs = surf(X_obs, Y_obs, Z_obs);
%     s_obs.FaceColor = 'black';
% end

% 绘制A车路径轨迹
plot(xled, yled, 'x', 'LineStyle', '--', 'Color', 'r');

% leading car (固定位置)
% pX_led = [-3 3 3 -3; 
%     -3 -3 3 3;
%     -3 -3 3 3;
%     -3 3 3 -3]';
% pY_led = [-1 -1 1 1;
%     -1 -1 -1 -1
%     1 1 1 1
%     -1 -1 1 1]';
% pZ_led = [0 0 0 0;
%     0 2 2 0
%     0 2 2 0
%     2 2 2 2]';
% p1 = patch(pX_led, pY_led, pZ_led, 'red');
% p1.FaceAlpha = 0.5;
 fig.Children.Projection = 'perspective';
 axis equal
% view(3)
 hold on;
% car B (6*2*2)
pX_B = [-3 3 3 -3; 
    -3 -3 3 3;
    -3 -3 3 3;
    -3 3 3 -3]';
pY_B = [-1 -1 1 1;
    -1 -1 -1 -1
    1 1 1 1
    -1 -1 1 1]';
pZ_B = [0 0 0 0;
    0 2 2 0
    0 2 2 0
    2 2 2 2]';
pB = patch(pX_B, pY_B, pZ_B, 'blue');
pB.FaceAlpha = 0.5;
view(3)
OrigVertsB = pB.Vertices; % 保持原始顶点

nx = timestep + 1;
cxB = zsaveB(1, :);
cyB = zsaveB(2, :);

vel_led = [xled' yled' zeros(nx, 1)];
vel_B = [cxB' cyB' zeros(nx, 1)];

for i = 1:nx 
    axis([0 70 -5 5 0 6]);
    pB.Vertices = OrigVertsB + repmat(vel_B(i,:), [size(OrigVertsB, 1), 1]);
    pause(2 * TS); 
end


%% Export the data to visualize
save('leading_car_state.mat', 'zref');
save('follower_carB_state.mat', 'zsaveB');
xlead = zref(1,:);
ylead = zref(2,:);
thetalead = zref(4,:);
xlead_interp = 0:.25:50;
ylead_interp = interpn(xlead,ylead,xlead_interp,'spline');
thetalead_interp = interpn(xlead,thetalead,xlead_interp,'spline');

% x_folB = zsaveB(1,:);
% y_folB = zsaveB(2,:);
% theta_folB = zsaveB(4,:);
% x_folB_interp = 0:.25:50;
% [x, index] = unique(x); 
% y_folB_interp = interpn(x_folB,y_folB,x_folB_interp,'spline');
% theta_folB_interp = interpn(x_folB,theta_folB,x_folB_interp,'spline');
%% write file
fileID = fopen('leading_car.txt','w');
fprintf(fileID,'positionXA:[');
for i = 1:size(xlead_interp,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',xlead_interp(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',xlead_interp(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',xlead_interp(size(xlead_interp,2)));
fprintf(fileID,'positionYA:[');
for i = 1:size(xlead_interp,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',ylead_interp(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',ylead_interp(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',ylead_interp(size(ylead_interp,2)));
fprintf(fileID,'thetaA:[');
for i = 1:size(thetalead_interp,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',thetalead_interp(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',thetalead_interp(i));
    end
   
end
fprintf(fileID,'%4.6f],',thetalead_interp(size(thetalead_interp,2)));

fclose(fileID);


%%
load('four obstacles MPC Test_Noa.mat');
steerlead = uOpt(1,:);

fileID = fopen('leading_car_input.txt','w');
fprintf(fileID,'steerA:[');
for i = 1:size(steerlead,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',steerlead(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',steerlead(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',steerlead(size(steerlead,2)));

% B car
steerB = usaveB(2,:);
fprintf(fileID,'steerB:[');
for i = 1:size(steerB,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',steerB(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',steerB(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',steerB(size(steerB,2)));

%%
%load('four obstacles MPC Test_Noa.mat');
xlead = xOpt(1,:);
ylead = xOpt(2,:);
vlead = xOpt(3,:);
thetalead = xOpt(4,:);

fileID = fopen('leading_car.txt','w');
fprintf(fileID,'positionXA:[');
for i = 1:size(xlead,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',xlead(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',xlead(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',xlead(size(xlead,2)));

fprintf(fileID,'positionYA:[');
for i = 1:size(ylead,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',ylead(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',ylead(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',ylead(size(ylead,2)));

fprintf(fileID,'thetaA:[');
for i = 1:size(thetalead,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',thetalead(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',thetalead(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',thetalead(size(thetalead,2)));

% B car
xB = zsaveB(1,:);
yB = zsaveB(2,:);
vB = zsaveB(3,:);
thetaB = zsaveB(4,:);

fprintf(fileID,'positionXB:[');
for i = 1:size(xB,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',xB(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',xB(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',xB(size(xB,2)));

fprintf(fileID,'positionYB:[');
for i = 1:size(yB,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',yB(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',yB(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',yB(size(yB,2)));

fprintf(fileID,'thetaB:[');
for i = 1:size(thetaB,2)
    if mod(i,8) ~= 0 
        fprintf(fileID,'%4.6f, ',thetaB(i));
    elseif mod(i,8) == 0
        fprintf(fileID, '%4.6f, \n',thetaB(i));
    end
   
end
fprintf(fileID,'%4.6f],\n\n',thetaB(size(thetaB,2)));


fclose(fileID);

%% generate plot
t = linspace(1,100,100);

figure;
plot(t,zsaveB(1,1:100))
legend("car B")

figure;
plot(t,zsaveB(2,1:100))
legend("car B")

figure;
plot(t,zsaveB(3,1:100))
legend("car B")

figure;
plot(t,zsaveB(4,1:100))
legend("car B")


