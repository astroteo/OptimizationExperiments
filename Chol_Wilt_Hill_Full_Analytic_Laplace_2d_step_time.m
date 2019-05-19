function   [Rho,Ni,DV,count_impulse_time,count_impulse_done]=Chol_Wilt_Hill_Full_Analytic_Laplace_2d_step_time(dt,t_span,X_0)

global W  alpha_v M_2d N_2d lambda_v l_max x_obs_2d step_time 

l_max=1;

disp('it is me')

[N_time,~]=size(t_span);

Rho=zeros(N_time,3);
Ni=zeros(N_time,3);

Rho(1,:)=X_0(1:3)';
Ni(1,:)=X_0(4:6)';

t=0;
tau=dt;

DV=zeros(N_time,3);

count_impulse_time=0;
count_impulse_done=0;
for i=2:1:N_time
    
t=t+i*dt;

r=Rho(i-1,1:2);
v=Ni(i-1,1:2);


%parameters recovering
m1=M_2d(1,1);
m2=M_2d(2,2);

n1=N_2d(1,1);
n2=N_2d(2,2);

x=r(1);
y=r(2);

x_obs=x_obs_2d(1);
y_obs=x_obs_2d(2);

lambda_1=lambda_v(1);
lambda_2=lambda_v(2);

% gradient analytic computation:

%phi_harm_2d(i,j)=lambda_1*exp(-lambda_2*(X_2d)'*N_2d*(X_2d));
f_harm_x=lambda_1*(2*n1*(-lambda_2)*(x/l_max)*exp((-lambda_2)*(n1*(x/l_max)^2+n2*(y/l_max)^2)));
f_harm_y=lambda_1*(2*n2*(-lambda_2)*(y/l_max)*exp((-lambda_2)*(n1*(x/l_max)^2+n2*(y/l_max)^2)));

%phi_2d(i,j)=1/2*(X_2d/L_max-X_obs_2d/L_max)'*M_2d*(X_2d/L_max-X_obs_2d/L_max);
f_pos_x=m1*(x/l_max-x_obs/l_max);
f_pos_y=m2*(y/l_max-y_obs/l_max);

%d/dx(f(x,y)+g(x,y))=df(x,y)/dx+dg(x,y)/dx 
%d/dy(f(x,y)+g(x,y))=df(x,y)/dy+dg(x,y)/dy
fx=f_harm_x+f_pos_x;
fy=f_harm_y+f_pos_y;

f=[fx;fy];

 if t-floor(t/step_time)*step_time==0
%     
   
 count_impulse_time=count_impulse_time+1;

if dot(f,v) >=0 
    
    disp('fight_gradient')

 K=1*alpha_v; 

V=-K*f/norm(f);

Ni_old=Ni(i-1,:);
Ni(i-1,:)=[V(1),V(2),Ni_old(3)];
DV(i-1,:)=Ni(i-1,:)-Ni_old;

count_impulse_done=count_impulse_done+1;



end

 end  

Ct=cos(W*tau);
St=sin(W*tau);

Rho_Ni_0=[Rho(i-1,:)';Ni(i-1,:)'];

PHI=[4-3*Ct       0      0       St/W          2/W*(1-Ct)          0;
    6*(St-W*tau)  1      0    -2/W*(1-Ct)    1/W*(4*St-3*W*tau)    0;
        0         0      Ct       0                0           1/W*St;
     3*W*St       0      0        Ct             2*St              0;
   -6*W*(1-Ct)    0      0      -2*St           4*Ct-3             0;
        0         0    -W*St      0                0              Ct ];
    
Rho_Ni=PHI*Rho_Ni_0;
    
Rho(i,:)=Rho_Ni(1:3)';
Ni(i,:)=Rho_Ni(4:6)';

end

