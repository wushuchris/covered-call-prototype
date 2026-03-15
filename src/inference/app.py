# FAST API server, locally hosted on port 8009. similar logic as the FAST HTML server
# we use this space to set up the routes (app.post, gets). and trigger the corresponding backtesting, daily function
# custom logger for this service as well
# activated by the if __main.py__ clause at the bottom, we are going to be running the dev server for now which is spawned by a Popen process by a mainn.py  trigger

# we should start with an inference call method and a backtesting method
# the request is using a specific data structure (shared between both services)
# the app function is in charge of captuirng it, but unpacking is the job of the handler (backtesting/daily)
# async all the way down once again.