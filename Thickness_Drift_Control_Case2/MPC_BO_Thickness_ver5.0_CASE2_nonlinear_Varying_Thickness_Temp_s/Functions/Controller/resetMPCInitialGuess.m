function c = resetMPCInitialGuess(c, flags)

if flags.mpcType == 1
    c.mpc.opti.set_initial(c.mpc.Xss, c.bounds.x_init)
    c.mpc.opti.set_initial(c.mpc.Uss, c.bounds.u_init)
    c.mpc.opti.set_initial(c.mpc.Yss, c.bounds.y_init)
    c.mpc.opti.set_initial(c.mpc.Ycss, c.bounds.yc_init)
end

for k = 1:c.mpc.N
    c.mpc.opti.set_initial(c.mpc.U{k}, c.bounds.u_init)
    if c.mpc.soft == 1
        c.mpc.opti.set_initial(c.mpc.E{k+1}, zeros(c.ny,1))
    end
    c.mpc.opti.set_initial(c.mpc.X{k+1}, c.bounds.x_init)
end

end
