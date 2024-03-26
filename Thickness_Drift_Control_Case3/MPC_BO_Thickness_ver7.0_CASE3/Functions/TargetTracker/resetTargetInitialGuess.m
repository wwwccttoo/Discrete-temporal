function c = resetTargetInitialGuess(c)

c.tt.opti.set_initial(c.tt.X, c.bounds.x_init)
c.tt.opti.set_initial(c.tt.U, c.bounds.u_init)
c.tt.opti.set_initial(c.tt.Y, c.bounds.y_init)
c.tt.opti.set_initial(c.tt.Yc, c.bounds.yc_init)

end
