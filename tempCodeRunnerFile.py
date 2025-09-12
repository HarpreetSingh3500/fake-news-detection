
if __name__ == '__main__':
    # PORT is used by services like Render
    port = int(os.environ.get('PORT', 5000))
    # '0.0.0.0' makes the app accessible on your local network and for Render
    app.run(host='0.0.0.0', port=port)

