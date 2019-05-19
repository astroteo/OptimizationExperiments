function   [score,traj]  = evaluateIndividual_raw(hiddenLayer1,hiddenLayer2,biasHiddenLayer1,biasHiddenLayer2,r0,v0)

global t_max t_prop r_goal v_goal 

goal = 0;
collision =0
DV_tot =0;
t=0;
i =1;
inBox =1

while t < t_max && goal ==0 && collision ==0 && inBox ==1
    
    DV = computeAction(hiddenLayer1,hiddenLayer2,biasHiddenLayer1,biasHiddenLayer2,r0,v0)
    DV_tot = DV_tot + norm(DV);
    [r,v] = propagateDynamic(r0, v0 + DV, t_prop);
    
    traj(i,:) = [r(1),r(2),0,v(1),v(2),0];
    i = i+1;
    collision = evaluateCollision(r);
    
    inBox = evaluateinBox(r)
    
    if norm(r-r_goal) < 2 && v-v_goal < 2
        goal = 1;
    end

    r0 = r;
    v0 = v;
    t = t + t_prop;
    

end

if collision
    disp('COLLISION')
end
    
if ~inBox
    disp('OUT OF BOX')
end
    
if goal
    disp('GOOOOOALLL')
end

if t >  t_max -t_prop
     disp('NOT ARRIVED')
end





if collision ~=0 && inBox ==1
    score = 0.3;
    if goal
        score = score + 0.3 + 1/DV_tot;
    end
else
    score =0;
end

end



function [r , v ] = propagateDynamic(r0,v0, dt)


X_0_3d = [r0(1);r0(2);0; v0(1);v0(2);0];


options = odeset('RelTol',1e-13,'AbsTol', 1e-12);
[~,x_CW_FD]= ode113('Chol_Wilt_Hill',dt,X_0_3d,options);%<--[]

r = x_CW_FD(end,1:2)';
v = x_CW_FD(end,1:2)';

end

function collision = evaluateCollision(x)

collision =0;

if norm(x) < 50
    collision =1;
end
 
end

function inBox = evaluateinBox(x)

inBox =0
global L

if norm(x) < L
    inBox =1;
end
 
end