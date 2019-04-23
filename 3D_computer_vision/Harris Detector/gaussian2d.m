function f=gaussian2d(sigma)
x = -ceil(3*sigma):ceil(3*sigma);
v = exp(-x.^2./(2*sigma^2));
f=v'*v;
f=f./sqrt(sum(f(:).^2));
end