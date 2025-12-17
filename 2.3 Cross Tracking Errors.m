zB=zsaveB;
error=zeros(101,1);
for i =1:98
    temperror=100;
    for j = 1:101
        temp=sqrt((zB(1,i)-zref(1,j))^2+(zB(2,i)-zref(2,j))^2);
        if temp<temperror
            temperror=temp;
        end
    end
    error(i)=temperror;
end
max(error)
sqrt(mse(error))