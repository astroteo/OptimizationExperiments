clear all
close all

%% ISS Orbit random chaser.

%orbit
Mu=(3.9856e+14);
r_Earth=6380*10^3;

h_orb=400*10^3;
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


%% desired obsrvation point & ISS ellipsoid
x_obs=0
y_obs=100-3
z_obs=0



lenght_ISS=50; %--> along y
width_ISS=50; %--> alng z
height_ISS=30;


[x_ISS, y_ISS, z_ISS] = ellipsoid(0,0,0,lenght_ISS,height_ISS,width_ISS);



%% Laplace 2D Potential 

X_obs_2d=[x_obs;y_obs];%-> observation point given in LVLH r.f.

%m1=lenght_ISS;
m1=1;

%m2=width_ISS;
m2=1;


M_2d=[m1, 0;
      0, m2];

flagx_obs=0;
flagy_obs=0;

L_max=800;
step_grid=1;
R1_bounded=-L_max:step_grid:(L_max-step_grid);

R1_boundedx=R1_bounded;
R1_boundedy=R1_bounded;

[~,N]=size(R1_boundedx);
[~,M]=size(R1_boundedy);

phi_2d=zeros(N,M);

setGlobaltotal_stepx(N);
setGlobaltotal_stepy(M); 

% plot potential
for i=1:N
    
    for j=1:M
        
       
            
            x=R1_boundedx(1,i)/L_max;
            y=R1_boundedy(1,j)/L_max;
           
            X_2d= [x;y];
            
           phi_2d(i,j)=1/2*(X_2d-X_obs_2d/L_max)'*M_2d*(X_2d-X_obs_2d/L_max);
           
          
            
            if y >= y_obs+step_grid && flagy_obs==0 
                
                j_obs=j-2;
                flagy_obs=1;
            end
            
            
            if x >=x_obs+step_grid && flagx_obs==0
                
                i_obs=i-2;
                flagx_obs=1;
            end
            
            
   
    end
end


step_gridx=step_grid;
step_gridy=step_grid;
[px,py] = gradient(phi_2d,step_gridy,step_gridx);




% figure()
% surf(x_ISS, y_ISS, z_ISS,'facecolor','g')
% hold on
% surf(R1_bounded,R1_bounded,phi_2d)
% grid on  
% 
% figure()
% contour(R1_bounded,R1_bounded,phi_2d)
% hold on
% surf(x_ISS, y_ISS,z_ISS,'facecolor','g')
% hold on
% plot3(x_obs,y_obs,0,'*b')
% hold on
% quiver(R1_bounded,R1_bounded,px,py)
% hold off

%% Initial Conditions & Integration time 

% !!!! CRUCIAL TEST: by  givinig initial conditions such that x_obs_2d is
% reached no trajectory corrections are made. ====> OK !!! **
Rho_0_2d=[100;-100];
Rho_0_3d=[Rho_0_2d(1);Rho_0_2d(2);0];

tau=1/4*T_orb;
Ct=cos(w*tau);
St=sin(w*tau);

X_obs_3d=[x_obs;y_obs;z_obs];

det=(4*St)/(w^3) -(8*Ct*St)/(w^3) +(4*Ct^2*St)/(w^3)+(4*St^3)/(w^3) -(3*St^2*tau)/(w^2);

N_tau_inv=1/det*[(4*St^2)/(w^2)-(3*St*tau)/w,     -((2*St)/(w^2))+(2*Ct*St)/(w^2),                        0;
                 (2*St)/(w^2)-(2*Ct*St)/(w^2),              St^2/(w^2),                                   0;
                           0,                                  0,              4/(w^2)-(8*Ct)/(w^2)+(4*Ct^2)/(w^2)+(4*St^2)/(w^2)-(3*St*tau)/w];
                   
M_tau=[-3*Ct+4,        0,   0;
        6*St-6*w*tau,  1,   0;                   
               0    ,  0,  Ct];


Ni_0_3d=N_tau_inv*(X_obs_3d-M_tau*Rho_0_3d);
Ni_0_2d_CT=[Ni_0_3d(1);Ni_0_3d(2)]; %** CRUCIAL TEST
Ni_0_2d_RAND=[-rand(1)*1e-1; -rand(1)*1e-1]; %% GENERIC STUFF
Ni_0_2d=Ni_0_2d_CT-Ni_0_2d_RAND

X_0_2d=[Rho_0_2d;Ni_0_2d];



%% Free Drift !! UNforced Problem!!

t_end=tau;
X_0_3d=[Rho_0_3d;Ni_0_3d];

options = odeset('RelTol',1e-13,'AbsTol', 1e-12);
[t_CW_FD,x_CW_FD]= ode113('Chol_Wilt_Hill',t_end,X_0_3d,options);%<--[]

[CW_FD,~]=size(t_CW_FD);
zebra_FD=zeros(CW_FD,1);

figure()
surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
hold on
plot3(x_CW_FD(:,1),x_CW_FD(:,2),zebra_FD,'--b',x_CW_FD(1,1),x_CW_FD(1,2),zebra_FD,'og',x_CW_FD(end,1),x_CW_FD(end,2),zebra_FD,'or',x_obs,y_obs,0,'*b')
grid on 


%% Test for Spherical Armonic.

X_coll_2d=zeros(2,1);

lambda_1_paper=1/2*(Rho_0_2d./L_max-X_coll_2d./L_max)'*(M_2d*(Rho_0_2d./L_max-X_coll_2d./L_max));
lambda_1_logic=1/2*([L_max;L_max]./L_max-X_coll_2d./L_max)'*(M_2d*([L_max;L_max]/L_max-X_coll_2d./L_max));
lambda_1_MY=max(max(phi_2d));
lambda_1=lambda_1_MY;
safety_radius=10;
lambda_2_MY=((lenght_ISS+safety_radius)/2)^2/L_max^2;
lambda_2_paper=1e-4;
lambda_2=lambda_2_MY;
n1=(lenght_ISS+safety_radius)/lenght_ISS;
n2=(width_ISS+safety_radius)/lenght_ISS;
N_2d=[n1,0;
      0, n2]; % spherical-shape of potential around obstacle.
  
%N_2d=eye(2);

phi_harm_2d=zeros(N,M);
phi_disc_2d=zeros(N,M);



for i=1:N
    
    for j=1:M
        
       
            
            x=R1_boundedx(1,i)/L_max;%--> necessity to adimensionalize
            y=R1_boundedy(1,j)/L_max;
           
            X_2d= [x;y];
            
           
           phi_harm_2d(i,j)=lambda_1*exp(-lambda_2*(X_2d)'*N_2d*(X_2d));
           
           if  x*L_max>(-lenght_ISS/2) &&  x*L_max<(lenght_ISS/2) && y*L_max>(-lenght_ISS/2) &&  y*L_max<(lenght_ISS/2)
               
               phi_disc_2d(i,j)=lambda_1;
               
           else
               phi_disc_2d(i,j)=0;
           end
    end
end

phi_2d=phi_harm_2d+phi_2d;
phi_disc_2d=phi_2d+phi_disc_2d;



[phi_min,j_min] = min((min(phi_2d)));
[~,i_min]=(min(phi_2d));
i_min=i_min(1)
j_min

disp_potentials=1;

if disp_potentials==1
    
figure()
mesh(R1_bounded,R1_bounded,phi_2d)
hold on
title('"position" potential')
grid on  

figure()
contour(R1_bounded,R1_bounded,phi_2d)
hold on
title('total potential contour')
plot3(x_obs,y_obs,0,'*b')
hold on

figure()
hold on
mesh(R1_bounded,R1_bounded,phi_2d)
title('total potential')
grid on  

figure()
mesh(R1_bounded,R1_bounded,phi_harm_2d)
title('harmonic potential')
grid on  

% figure()
% mesh(R1_bounded,R1_bounded,phi_disc_2d)
% title('Phi_disc+Phi_pos')
% grid on  
% 
% figure()
% mesh(R1_bounded,R1_bounded,phi_disc_2d)
% title('Phi_disc+Phi_pos contour')
% grid on  

figure()
contour(R1_bounded,R1_bounded,phi_harm_2d)
hold on
plot3(x_obs,y_obs,0,'*b',Rho_0_2d(1),Rho_0_2d(2),0,'ob')
title('Phi_harm contour') 
hold on
% quiver(R1_bounded,R1_bounded,px,py)
% hold off

end

%% GNC by Laplace Aritificial Potential

% %parameters to compute discrete gradient
% setGlobalstep_grid(step_grid);
% setGlobalphi_2d(phi_2d );
% setGlobalx_obs_2d(X_obs_2d);
% 
% 
% 
% 
% % alpha IsSue
% alpha_v=norm(Rho_0_2d-X_obs_2d)/t_end
% % alpha_v=norm(Ni_0_2d)*1e-4;
% setGlobalalpha_v(alpha_v );
% t_end_Lap=t_end+1.5*T_orb
% %integration
% options = odeset('RelTol',1e-6,'AbsTol', 1e0);
% [t_CW,x_CW]= ode113('Chol_Wilt_Hill_Laplace_2d',t_end_Lap,X_0_2d,options);%<--[]
% 
% %step_time to reduce control action
% step_time=50;
% setGlobalstep_time(step_time)
% flag_time=0;
% setGlobalflag_time(flag_time )
% 
% % options = odeset('RelTol',1e-13,'AbsTol', 1e-3);
% % [t_CW,x_CW]= ode113('Chol_Wilt_Hill_Laplace_2d_time_flag',t_end_Lap,X_0_2d,options);%<--[]
% 
% 
% [CW,~]=size(t_CW);
% zebra=zeros(CW,1);
% 
% 
% %plot
% figure()
% surf(x_ISS, y_ISS,zeros(size(x_ISS)),'facecolor','g')
% hold on
% plot3(x_CW(:,1),x_CW(:,2),zebra,'--b',x_CW(1,1),x_CW(1,2),zebra,'og',x_CW(end,1),x_CW(end,2),zebra,'or',x_obs,y_obs,0,'*b')
% grid on 




