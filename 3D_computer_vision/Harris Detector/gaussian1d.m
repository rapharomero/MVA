
function f = gaussian1d(sigma)
x = -ceil(3*sigma):ceil(3*sigma);
f = exp(-x.^2./(2*sigma^2));
f=f./sqrt(sum(f(:).^2));
end