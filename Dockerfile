# Use an official Node.js image as the base image for our build environment.
# We specify the version to ensure consistency.
FROM node:18-alpine AS builder

# Set the working directory inside the container.
WORKDIR /app

# Copy the package.json and package-lock.json files first.
# This leverages Docker's layer caching. If these files don't change,
# Docker won't re-run 'npm install' in subsequent builds, speeding up the process.
COPY package.json package-lock.json ./

# Install all the project dependencies.
RUN npm install

# Copy the rest of the application's source code into the container.
COPY . .

# Build the React application for production.
# This creates a 'dist' folder with optimized static files (HTML, CSS, JS).
RUN npm run build
# CMD ["sleep", "600"] 
# --- Stage 2: The Production Server ---
# Use a lightweight Nginx image as the base for our production environment.
# 'alpine' versions are very small and secure.
FROM nginx:1.25-alpine

# Copy the built static files from the 'builder' stage into the Nginx server's
# default directory for serving web content.
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx will automatically look for and serve 'index.html' from this directory.
# No need to specify a CMD as the base Nginx image already has one.
# Expose port 80 to allow traffic to the Nginx server.
