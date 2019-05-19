function   DV = computeAction(hiddenLayer1,hiddenLayer2,biasHiddenLayer1,biasHiddenLayer2,r,v)

global L V alpha_v

x = [r./L;v./V];
%disp(x)

%h1 = sigmf(hiddenLayer1 * x + biasHiddenLayer1,[1,0]);
h1 = tanh(hiddenLayer1 * x + biasHiddenLayer1);%%%muhc
h2 = hiddenLayer2 * h1 + biasHiddenLayer2;

DV_ =  V * h2 ;
DV = zeros(2,1);

for i=1:2
    if DV_(i) >= 0  

        if DV_(i) < alpha_v
            DV(i) = DV(i);
        else
            DV(i) = alpha_v;
        end

    else 

        if DV_(i) > - alpha_v
            DV(i) = DV(i);
        else
            DV(i) = -alpha_v;
        end

    end
end


end