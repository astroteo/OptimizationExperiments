clear all
close all

%% ISS Orbit random chaser.

%orbit
Mu=(3.9856e+14);
r_Earth=6380*10^3;

h_orb=250*10^3;
i_orb=51.65*pi/180;
%i_orb=0;
om_orb=0;
OM_orb=0;
theta0_orb=0;
a_orb=r_Earth+h_orb;
e_orb=0;


T_orb=2*pi*sqrt((r_Earth+h_orb)^3/Mu)

omega=sqrt(Mu/((r_Earth+h_orb)^3));
w=omega;
setGlobalW(w);


%target:
[R_0t,V_0t]=OrbPar2RV_MY(a_orb,e_orb,i_orb,OM_orb,om_orb,theta0_orb,Mu);
w_v=cross(R_0t,V_0t)/(norm(R_0t)^2);

%chaser:
R_0c=(R_0t+rand(3,1)*0.25*1e2); 
V_0c=(V_0t+rand(3,1)*0.5*1e-1);

 ROT_plane0=[cos(om_orb)*cos(OM_orb)-sin(om_orb)*cos(i_orb)*sin(OM_orb) -sin(om_orb)*cos(OM_orb)-cos(om_orb)*cos(i_orb)*sin(OM_orb) sin(i_orb)*sin(OM_orb)
            cos(om_orb)*sin(OM_orb)+sin(om_orb)*cos(i_orb)*cos(OM_orb) -sin(om_orb)*sin(OM_orb)+cos(om_orb)*cos(i_orb)*cos(OM_orb) -sin(i_orb)*cos(OM_orb)
              sin(om_orb)*sin(i_orb)                                                 cos(om_orb)*sin(i_orb)                                 cos(i_orb)]  ;


%% Desired Observation Point & Obstacle (ISS ellipsoid)
x_obs=13
y_obs=70-3
z_obs=0

X_obs_2d=[x_obs;y_obs];
X_obs_3d=[x_obs;y_obs;z_obs];

lenght_ISS=40; %--> along y
width_ISS=50; %--> alng z
height_ISS=40;


[x_ISS, y_ISS, z_ISS] = ellipsoid(0,0,0,lenght_ISS,height_ISS,width_ISS);

%% Initial Conditions & Integration time 

% !!!! CRUCIAL TEST: by  givinig initial conditions such that x_obs_2d is
% reached no trajectory corrections are made. ====> OK !!! **
Rho_0_2d=[5;-75];
Rho_0_3d=[Rho_0_2d(1);Rho_0_2d(2);0];

tau=1/4*T_orb;
Ct=cos(w*tau);
St=sin(w*tau);



det=(4*St)/(w^3) -(8*Ct*St)/(w^3) +(4*Ct^2*St)/(w^3)+(4*St^3)/(w^3) -(3*St^2*tau)/(w^2);

N_tau_inv=1/det*[(4*St^2)/(w^2)-(3*St*tau)/w,     -((2*St)/(w^2))+(2*Ct*St)/(w^2),                        0;
                 (2*St)/(w^2)-(2*Ct*St)/(w^2),              St^2/(w^2),                                   0;
                           0,                                  0,              4/(w^2)-(8*Ct)/(w^2)+(4*Ct^2)/(w^2)+(4*St^2)/(w^2)-(3*St*tau)/w];
                   
M_tau=[-3*Ct+4,        0,   0;
        6*St-6*w*tau,  1,   0;                   
               0    ,  0,  Ct];


Ni_0_3d=N_tau_inv*(X_obs_3d-M_tau*Rho_0_3d);
Ni_0_2d_CT=[-Ni_0_3d(1);-Ni_0_3d(2)]; %** CRUCIAL TEST
Ni_0_2d_RAND=[-rand(1)*1e-3; -rand(1)*1e-3]; %% GENERIC STUFF
Ni_0_2d=Ni_0_2d_CT-Ni_0_2d_RAND;

X_0_2d=[Rho_0_2d;Ni_0_2d];


%% Free Drift !! UNforced Problem!!

t_end=2*tau;

Static=1;

if Static==0
    
X_0_3d=[Rho_0_3d;Ni_0_3d];

else
    
X_0_3d=[Rho_0_3d;zeros(3,1)];

end


options = odeset('RelTol',1e-13,'AbsTol', 1e-12);
[t_CW_FD,x_CW_FD]= ode113('Chol_Wilt_Hill',t_end,X_0_3d,options);%<--[]

[CW_FD,~]=size(t_CW_FD);
zebra_FD=zeros(CW_FD,1);

figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(x_CW_FD(:,1),x_CW_FD(:,2),zebra_FD,'--b',x_CW_FD(1,1),x_CW_FD(1,2),zebra_FD,'og',x_CW_FD(end,1),x_CW_FD(end,2),zebra_FD,'or',x_obs,y_obs,0,'*b')
hold on
title('free drift')
axis equal
grid on 

%% Grid Properties 

L_max=2 * norm(Rho_0_2d);  

step_grid=5;

R1_bounded=-L_max:step_grid:(L_max-step_grid);
    
R1_boundedx=R1_bounded;
R1_boundedy=R1_bounded;

[~,N]=size(R1_boundedx);
[~,M]=size(R1_boundedy);



%% Position Potential PARAMETERS


m1=1;


m2=1;


M_2d=0.005*[m1, 0;
            0, m2];

%% Position Potential
X_obs_2d=[x_obs;y_obs];%-> observation point given in LVLH r.f.

phi_pos_2d=zeros(N,M);

for i=1:N
    
    for j=1:M
        
       
            
            x=R1_boundedx(1,i);
            y=R1_boundedy(1,j);
           
            X_2d= [x;y];
            
           phi_pos_2d(i,j)=1/2*(X_2d-X_obs_2d)'*(M_2d*(X_2d-X_obs_2d));
            
            
   
    end
end


%% Spherical Harmonic 2D Potential PARAMETERS
safety_radius=0;

n1=(lenght_ISS+safety_radius)/lenght_ISS;
n2=(width_ISS+safety_radius)/lenght_ISS;
n3=(height_ISS+safety_radius)/lenght_ISS;

N_2d=0.008*[n1,0;
            0, n2] % spherical-shape of potential around obstacle.
  

lambda_2=1;%--> since potential is defined in the old way      
phi_pos_2d_0=1/2*(Rho_0_2d-X_obs_2d)'*M_2d*(Rho_0_2d-X_obs_2d);

R_bound_2d=[lenght_ISS;width_ISS];
X_obj_2d=[0;0];
phi_pos_2d_bound=1/2*(R_bound_2d-X_obj_2d)'*M_2d*(R_bound_2d-X_obj_2d);

phi_harm_2d_bound=1/2*exp((R_bound_2d-X_obj_2d)'*N_2d*(R_bound_2d-X_obj_2d));

%lambda_1=1e0*(phi_pos_2d_0-phi_pos_2d_bound)/phi_harm_2d_bound ; %<-- phi_att(x_bound)+phi_rep(x_bound)=
%lambda_1=1e5;
lambda_1=((phi_pos_2d_0)+1/2*norm(Ni_0_2d)^2)/exp(-lambda_2*(R_bound_2d-X_obj_2d)'*N_2d*(R_bound_2d-X_obj_2d))/10000000000000000;


lambda_v=[lambda_1;lambda_2];


%% Harmonic Potential

phi_harm_2d=zeros(N,M);
X_coll_2d=zeros(2,1);

for i=1:N
    
    for j=1:M
        
       
            
            x=R1_boundedx(1,i);%--> necessity to adimensionalize
            y=R1_boundedy(1,j);
           
            X_2d= [x;y];
            
           
           phi_harm_2d(i,j)=1/2*lambda_1*exp(-lambda_2*(X_2d-X_coll_2d)'*N_2d*(X_2d-X_coll_2d));
           %phiiii=phi_harm_2d(i,j)
           
           
    end
end


%% Total Potential & Plots

phi_2d=phi_harm_2d+phi_pos_2d;

R1_boundedx=R1_bounded;
R1_boundedy=R1_bounded;

phi_min=max(max(phi_2d));

for i=1:N
    for j=1:M
        
        if phi_2d(i,j) < phi_min
            
            phi_min=phi_2d(i,j);
            i_min=i;
            j_min=j;
        end
        
    end
end

x_min=R1_boundedx(i_min);
y_min=R1_boundedx(j_min);

Disp_Potentials=1;
if Disp_Potentials==1

figure()
mesh(R1_boundedx,R1_boundedy,phi_pos_2d)
hold on
title('"position" potential')
grid on  

figure()
contour(R1_boundedx,R1_boundedy,phi_pos_2d)
hold on
plot3(x_obs,y_obs,0,'*b',Rho_0_2d(1),Rho_0_2d(2),0,'ob')
title('Phi_{pos} contour') 
hold on

figure()
mesh(R1_boundedx,R1_boundedy,phi_harm_2d)
title('harmonic potential')
grid on  

figure()
contour(R1_boundedx,R1_boundedy,phi_harm_2d)
hold on
plot3(x_obs,y_obs,0,'*b',Rho_0_2d(1),Rho_0_2d(2),0,'ob')
title('Phi_{harm} contour') 
hold on



figure()
contour(R1_boundedx,R1_boundedy,phi_2d')
hold on
title('total potential contour')
plot3(x_obs,y_obs,0,'*b',x_min,y_min,0,'*r')
hold on

figure()
hold on
mesh(R1_boundedx,R1_boundedy,phi_2d')
title('total potential')
grid on  

% quiver(R1_bounded,R1_bounded,px,py)
% hold off
end






%% GNC by Laplace Aritificial Potential [ANALYCAL GRADIENT]

%parameters to compute analytical gradient
setGloball_max(L_max)
setGlobalM_2d(M_2d)
setGlobalN_2d(N_2d)
setGloballambda_v(lambda_v)
setGlobalx_obs_2d(X_obs_2d )


% alpha IsSue
%alpha_v=norm(Rho_0_2d-X_obs_2d)/t_end
f_sat=1e-2;
m_cubesat=5;
alpha_v=f_sat/m_cubesat
% alpha_v=norm(Ni_0_2d)*1e-4;
setGlobalalpha_v(alpha_v );

Perform_Integration=0;
if Perform_Integration==1
%integration time
t_end_Lap=t_end+1.5*T_orb

%integration
options = odeset('RelTol',1e-7,'AbsTol', 1e-1);
[t_CW,x_CW]= ode113('Chol_Wilt_Hill_Laplace_2d_analytic',t_end_Lap,X_0_2d,options);%<--[ISSUE: n° of correction strictly dependant on ODE's precision]


%plot
[CW,~]=size(t_CW);
zebra=zeros(CW,1);

figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(x_CW(:,1),x_CW(:,2),zebra,'--b',x_CW(1,1),x_CW(1,2),zebra,'og',x_CW(end,1),x_CW(end,2),zebra,'or',x_obs,y_obs,0,'*b')
title('collision Avoidance with analytical gradient')
grid on 

end


%% GNC by Artificial potential: Parameters globalization:

%parameters to compute analytical gradient/hessian
setGloball_max(L_max)
setGlobalM_2d(M_2d)
setGlobalN_2d(N_2d)
setGloballambda_v(lambda_v)
setGlobalx_obs_2d(X_obs_2d )

%step_time to reduce control action
step_time=100
setGlobalstep_time(step_time)
flag_time=0;
setGlobalflag_time(flag_time )
time_before=0;
setGlobaltime_before(time_before)
time_last_fire=0;
setGlobaltime_last_fire(time_last_fire)



% alpha IsSue
alpha_v=norm(Rho_0_2d-X_obs_2d)/(1*t_end) % alpha_v=norm(Ni_0_2d)*1e-4;
setGlobalalpha_v(alpha_v );

%% Continous control
options = odeset('RelTol',1e-6,'AbsTol', 1e-6);
[t_CW_FC,x_CW_FC]= ode113('Chol_Wilt_Hill_Continous_full_analytic',0.02*tau,X_0_3d,options);
zebra_FC =zeros(size(t_CW_FC));
figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(x_CW_FC(:,1),x_CW_FC(:,2),zebra_FC,'--b',x_CW_FC(1,1),x_CW_FC(1,2),zebra_FC,'og',x_CW_FC(end,1),x_CW_FC(end,2),zebra_FC,'or',x_obs,y_obs,0,'*b')
title('collision Avoidance with analytical gradient => Force Control')
grid on 

%% Full Analytical Computation: Velocity contorl [discrete time control action]

dt= 10;
t_span=[0:dt:5*tau]';

[Rho_an,Ni_an,DV_v,count_impulse_time,count_impulse_done]=Chol_Wilt_Hill_Full_Analytic_Laplace_2d_step_time(dt,t_span,X_0_3d);

count_impulse_time
count_impulse_done
[size_ODE,~]=size(t_CW_FD)
%[size_ODE_ctrl,~]=size(t_CW)
[size_SPAN,~]=size(t_span)

figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(Rho_an(:,1),Rho_an(:,2),Rho_an(:,3),'--b',Rho_an(1,1),Rho_an(1,2),Rho_an(1,3),'og',Rho_an(end,1),Rho_an(end,2),Rho_an(end,3),'or',x_obs,y_obs,0,'*b')
title('collision Avoidance with analytical gradient & analytical trajectory propagation')
grid on 
axis equal
Set_Point_traj_2=[t_span,Rho_an,Ni_an];

%save('Set_Point_traj_2');

%% Test NN to compute an action
V_max = alpha_v;
setGlobalLV(L_max,V_max)

n_hidden1 = 16;
n_hidden2 = n_hidden1;

hiddenLayer1 = rand(n_hidden1,4) *2 - ones(n_hidden1,4)
biasHiddenLayer1 = ones(n_hidden1,1);

hiddenLayer2 = rand(2,n_hidden2) * 2 - ones(2,n_hidden2)
biasHiddenLayer2 = ones(2,1);

v_goal_ = [0;0]
r_goal_ = X_obs_2d
setGlobalrv_goal(r_goal_,v_goal_)

%test the dimensions:
h1 = hiddenLayer1 * [-Rho_0_2d./L_max;Ni_0_2d./V_max] + biasHiddenLayer1
out = V_max* (hiddenLayer2 * h1 + biasHiddenLayer2)
DV_test = computeAction(hiddenLayer1,hiddenLayer2,biasHiddenLayer1,biasHiddenLayer2,Rho_0_2d,Ni_0_2d)


%gloalize propagation time-step and maximum time to reach the goal
t_max_ = 2 * tau;%same as for APF;
t_prop_ = 10;%same as for APF;
setGlobalTmaxTprop(t_max_, t_prop_)

% test random initial individual
%[score,traj]  = evaluateIndividual_raw(hiddenLayer1,hiddenLayer2,biasHiddenLayer1,biasHiddenLayer2,Rho_0_2d,Ni_0_2d)
%figure()
%surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
%hold on
%plot3(traj(:,1),traj(:,2),traj(:,3),'--b',traj(1,1),traj(1,2),traj(1,3),'og',traj(end,1),traj(end,2),traj(end,3),'or',x_obs,y_obs,0,'*b')
%title('Random initial individual')
%grid on 
%axis equal


%% Test PopulationGeneration , classDef , PopulationEvaluation
n_individuals = 200
interval_weights = [-1,1]

r0 = Rho_0_2d;
v0 = Ni_0_2d;

individuals = generateFirstPopulation(n_individuals, n_hidden1,n_hidden2,interval_weights,r0,v0);
individualBestEver = individuals(1);
%save('individualBestEver.mat','individualBestEver');
individualsBest = computePerfPopulation(individuals)
%for i =individualsBest
    %i.score
    %i.traj
%end

%% Genetic algorithm optimization ==> Test
best_samples = round(n_individuals * 0.2);
lucky_few = round(n_individuals * 0.2);

individualsBreed = selectFromPopulation(individualsBest,best_samples, lucky_few)
figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(individualsBest(1).traj(:,1),individualsBest(1).traj(:,2),individualsBest(1).traj(:,3),'--b',...
    individualsBest(1).traj(1,1),individualsBest(1).traj(1,2),individualsBest(1).traj(1,3),'og',...
    individualsBest(1).traj(end,1),individualsBest(1).traj(end,2),individualsBest(1).traj(end,3),...
    'or',x_obs,y_obs,0,'*b')
title('Best among Random  individuals')
grid on 
axis equal

figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(individualsBest(end).traj(:,1),individualsBest(end).traj(:,2),individualsBest(end).traj(:,3),'--b',...
      individualsBest(end).traj(1,1),individualsBest(end).traj(1,2),individualsBest(end).traj(1,3),'og',...
      individualsBest(end).traj(end,1),individualsBest(end).traj(end,2),individualsBest(end).traj(end,3),...
      'or',x_obs,y_obs,0,'*b')
title('Worst among Random individuals')
grid on 
axis equal
%% Genetic algorithm optimization ==> True

number_of_child = round(n_individuals * 0.05);
chance_of_mutation = round(n_individuals * 0.02);

best_samples = round(n_individuals * 0.35);
lucky_few = round(n_individuals * 0.05);

population = generateFirstPopulation(n_individuals, n_hidden1,n_hidden2,interval_weights,r0,v0);


generation = 10;
% SUPER-KEY-POINT: (best_sample + lucky_few) / 2 * number_of_child = size_population
while(generation > 1)
    disp('new-generation')
    disp(generation)
    populationNext = createNextGeneration(population, best_samples, lucky_few,number_of_child,chance_of_mutation);
    disp('wtf:')
    size(populationNext)
    population = populationNext;
    disp('wtf:')
    size(population)
 
    generation = generation -1;
    
end

individualsBest = computePerfPopulation(populationNext);

figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(individualsBest(1).traj(:,1),individualsBest(1).traj(:,2),individualsBest(1).traj(:,3),'--b',...
    individualsBest(1).traj(1,1),individualsBest(1).traj(1,2),individualsBest(1).traj(1,3),'og',...
    individualsBest(1).traj(end,1),individualsBest(1).traj(end,2),individualsBest(1).traj(end,3),...
    'or',x_obs,y_obs,0,'*b')
title('Best among Optimized  individuals')
grid on 
axis equal

figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(individualsBest(end).traj(:,1),individualsBest(end).traj(:,2),individualsBest(end).traj(:,3),'--b',...
      individualsBest(end).traj(1,1),individualsBest(end).traj(1,2),individualsBest(end).traj(1,3),'og',...
      individualsBest(end).traj(end,1),individualsBest(end).traj(end,2),individualsBest(end).traj(end,3),...
      'or',x_obs,y_obs,0,'*b')
title('Worst among Optimized individuals')
grid on 
axis equal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SEE IF STARTING FROM ANOTHER POINT COLLISION IS AVOIDED
 
individualB = individualsBest(1);
individualB.r0 = (-1 + 2* rand ) *[ 0;L_max] %(individualsBest(1).r0;
individualB.v0 = (-1 + 2* rand )* individualsBest(1).v0;
%individualB.r0 = -individualsBest(1).r0;
%individualB.v0 = individualsBest(1).v0;
individualB.evaluate

individualB
 
figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(individualB(end).traj(:,1),individualB(end).traj(:,2),individualB(end).traj(:,3),'--b',...
      individualB(end).traj(1,1),individualB(end).traj(1,2),individualB(end).traj(1,3),'og',...
      individualB(end).traj(end,1),individualB(end).traj(end,2),individualB(end).traj(end,3),...
      'or',x_obs,y_obs,0,'*b')
title('Best among Optimized individuals == > RANDOM CI ')
grid on 
axis equal
