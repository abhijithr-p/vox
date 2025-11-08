<?php
session_start();
$conn = new mysqli("localhost", "root", "", "website_login");

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Check if user is logged in
if (!isset($_SESSION['user_id'])) {
    header("Location: login.html");
    exit();
}

$user_id = $_SESSION['user_id'];

$full_name = $conn->real_escape_string($_POST['full_name']);
$age = intval($_POST['age']);
$gender = $conn->real_escape_string($_POST['gender']);
$bio = $conn->real_escape_string($_POST['bio']);

$profile_picture = null;
if ($_FILES['profile_picture']['size'] > 0) {
    $target_dir = "uploads/";
    $profile_picture = $target_dir . basename($_FILES["profile_picture"]["name"]);
    move_uploaded_file($_FILES["profile_picture"]["tmp_name"], $profile_picture);
}

// Update query
$sql = "UPDATE users SET full_name='$full_name', age=$age, gender='$gender', bio='$bio'";

if ($profile_picture) {
    $sql .= ", profile_picture='$profile_picture'";
}

$sql .= " WHERE id='$user_id'";

if ($conn->query($sql) === TRUE) {
    echo "Profile updated successfully! <a href='profile.html'>Go back</a>";
} else {
    echo "Error updating profile: " . $conn->error;
}

$conn->close();
?>
