function [r,v]=OrbPar2RV_My(a,e,i,OM,om,theta,mu)

i=i*pi/180;
OM=OM*pi/180;
om=om*pi/180;



r_pe=(a*(1-e^2))/(1+e*cos(theta));
r_PE=[r_pe*cos(theta) r_pe*sin(theta) 0]';
p=a*(1-e^2);

v_PE=[-sqrt(mu/p)*sin(theta) sqrt(mu/p)*(e+cos(theta)) 0]';



ROT=[cos(om)*cos(OM)-sin(om)*cos(i)*sin(OM) -sin(om)*cos(OM)-cos(om)*cos(i)*sin(OM) sin(i)*sin(OM)
    cos(om)*sin(OM)+sin(om)*cos(i)*cos(OM) -sin(om)*sin(OM)+cos(om)*cos(i)*cos(OM) -sin(i)*cos(OM)
    sin(om)*sin(i) cos(om)*sin(i) cos(i)];   
    
r=ROT*r_PE;
v=ROT*v_PE;