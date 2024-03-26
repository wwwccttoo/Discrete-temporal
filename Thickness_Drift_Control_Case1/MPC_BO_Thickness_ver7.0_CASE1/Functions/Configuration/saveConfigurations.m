function saveConfigurations(c, flags, name)

warning('off')
save(name, 'c', 'flags')
warning('on')

end
