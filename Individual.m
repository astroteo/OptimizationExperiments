classdef Individual < handle
   properties (Access = public) 
      score
      traj
      r0
      v0
      hiddenLayer1
      biasHiddenLayer1
      hiddenLayer2
      biasHiddenLayer2
   end
   methods 
       function evaluate(obj)
           
           [obj.score,obj.traj] =  evaluateIndividual( obj.hiddenLayer1,obj.hiddenLayer2,...
                                                       obj.biasHiddenLayer1,obj.biasHiddenLayer2,...
                                                       obj.r0,obj.v0);
          
           
       end
   end
end


function   [score,traj]  = evaluateIndividual(hiddenLayer1,hiddenLayer2,biasHiddenLayer1,biasHiddenLayer2,r0,v0)

global t_max t_prop r_goal v_goal L V

goal = 0;
collision =0;
DV_tot =0;
t=0;
i =1;
inBox =1;
traj(1,:) = [r0(1),r0(2),0,v0(1),v0(2),0];

while t < t_max && goal ==0 && collision ==0 && inBox ==1
    
    DV = computeAction(hiddenLayer1,hiddenLayer2,biasHiddenLayer1,biasHiddenLayer2,r0,v0);
    %DV_tot = DV_tot + norm(DV - v0);
    %[r,v] = propagateDynamic(r0,  DV, t_prop);
    
    DV_tot =DV_tot + norm(DV);
    [r,v] = propagateDynamic(r0,v0+ DV, t_prop);
    
    i = i+1;
    traj(i,:) = [r(1),r(2),0,v(1),v(2),0];
    
    collision = evaluateCollision(r);
    
    inBox = evaluateinBox(r);
    
    if norm(r-r_goal) < 5
        goal = 1;
    end

    r0 = r;
    v0 = v;
    t = t + t_prop;
    

end

verbose = 0;
if verbose 
    if collision
        disp('COLLISION')
    end

    if ~inBox
        disp('OUT OF BOX')
        disp(r)
    else
        if goal
        disp('GOOOOOALLL')
        else
        disp('NOT ARRIVED')
        disp(r)
        end
    end
end



if collision == 0 
    %score = 0.5 * (0.9 *(1/(norm((r - r_goal)./L)) + 0.1 * 1/(norm(v./V- v_goal./V))));
    score = - norm((r - r_goal)./L ); %- norm(v./V- v_goal./V);
    %if inBox
        %score = score +  1/(DV_tot/V);
    %else
        %score = score - norm((r - r_goal)./L );
    %end
    %else
        %score = score - norm((r-r_goal)./L);
    %end
    
    if goal
        score = score + 1.25 + 1/(DV_tot/V);
    end
else
    score =  - norm((r - r_goal)./L )- (norm(v./V));%- norm((r - r_goal)./L )
end

end



function [r , v ] = propagateDynamic(r0,v0, dt)


X_0_3d = [r0(1);r0(2);0; v0(1);v0(2);0];

options = odeset('RelTol',1e-13,'AbsTol', 1e-12);
[~,x_CW_FD]= ode113('Chol_Wilt_Hill',dt,X_0_3d,options);%<--[]

r = x_CW_FD(end,1:2)';
v = x_CW_FD(end,4:5)';

end

function collision = evaluateCollision(x)

collision =0;

if norm(x) < 40
    collision =1;
end
 
end

function inBox = evaluateinBox(x)

inBox =0;
global L

if norm(x) < L
    inBox =1;
end
 
end