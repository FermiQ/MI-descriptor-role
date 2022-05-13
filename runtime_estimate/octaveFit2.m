[estim, fssize, rung, sis, dim, tfs, tfit] = textread("runtimesServerMod.txt", "%d %d %d %d %d %f %f", 'headerlines', 1);

X = log10(estim);
Y = log10(fssize);
[x, i] = sort(X);
y = Y(i);

Ftem = @(x) ([ones(length(x), 1), x, x.**2, x.**3, x.**4]);
F = feval(Ftem, x);
%F = [ones(n, 1), x(:), x.*log(x)];
[p, e_var, r, p_var, fit_var] = LinearRegression (F, y);
p
yFit = F * p;

Xextra = [1 : 0.1 : 13]';
Fextra = feval(Ftem, Xextra);
yextra = Fextra * p;

figure ()
plot(x, y, '+b', x, yFit, '-g', Xextra, yextra, '-r')
